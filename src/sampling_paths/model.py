from typing import Literal, overload, Any, Protocol, runtime_checkable

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from differt.geometry import normalize
from differt.scene import (
    TriangleScene,
)
from jaxtyping import (
    Array,
    ArrayLike,
    Float,
    Int,
    PRNGKeyArray,
)

from .metrics import reward_fn
from .submodels import Flows, ObjectsEncoder, SceneEncoder, StateEncoder
from .utils import geometric_transformation, unpack_scene


@runtime_checkable
class RewardFn(Protocol):
    """Protocol for a function that rewards a path candidate."""

    def __call__(
        self, path_candidate: Int[Array, " order"], scene: TriangleScene, **kwargs: Any
    ) -> Float[Array, ""]:
        """
        Compute the reward for a given path candidate in a scene.

        Args:
            path_candidate: The path candidate to be rewarded.
            scene: The scene in which the path candidate exists.
            kwargs: Additional keyword arguments.

        Returns:
            The reward value for the given path candidate in the scene.

        """
        ...


class Model(eqx.Module):
    """Model to sample path candidates in a scene."""

    # Static
    order: int = eqx.field(static=True)
    action_masking: bool = eqx.field(static=True)
    distance_based_weighting: bool = eqx.field(static=True)
    # Trainable
    objects_encoder: ObjectsEncoder
    scene_encoder: SceneEncoder
    state_encoder: StateEncoder
    flows: Flows
    # Training
    epsilon: Float[Array, ""]
    # Static but can be changed
    inference: bool
    reward: RewardFn

    def __init__(
        self,
        *,
        order: int,
        num_embeddings: int,
        width_size: int,
        depth: int,
        num_vertices_per_object: int = 3,
        dropout_rate: float = 0.0,
        epsilon: Float[ArrayLike, ""] = 0.5,
        action_masking: bool = False,
        distance_based_weighting: bool = False,
        inference: bool = False,
        reward_fn: RewardFn = reward_fn,
        key: PRNGKeyArray,
    ) -> None:
        """
        Initialize the model.

        Args:
            order: The order (length) of the path candidates to be sampled.
            num_embeddings: Number of embeddings for objects, scene, and state.
            width_size: Width of the hidden layers in the MLPs.
            depth: Number of hidden layers in the MLPs.
            num_vertices_per_object: Number of vertices per object (default is 3 for triangles).
            dropout_rate: Dropout rate to be used in the flows model.
            epsilon: Epsilon value for epsilon-greedy uniform sampling.
            action_pruning: Whether to use action pruning based on geometric considerations.
            distance_based_weighting: Whether to weight flows based on distances between objects.
            inference: Whether to run in inference mode (disables epsilon-greedy uniform sampling and dropout).
            reward_fn: The reward function to be used.
            key: The random key to be used.

        """
        self.order = order
        self.action_masking = action_masking
        self.distance_based_weighting = distance_based_weighting

        self.objects_encoder = ObjectsEncoder(
            num_embeddings=num_embeddings,
            width_size=width_size,
            depth=depth,
            num_vertices_per_object=num_vertices_per_object,
            key=key,
        )
        self.scene_encoder = SceneEncoder(
            num_embeddings=num_embeddings,
            width_size=width_size,
            depth=depth,
            key=key,
        )
        self.state_encoder = StateEncoder(
            order=order,
            num_embeddings=num_embeddings,
            key=key,
        )
        self.flows = Flows(
            in_size=self.objects_encoder.out_size
            + self.scene_encoder.out_size
            + self.state_encoder.out_size,
            width_size=width_size,
            depth=depth,
            dropout_rate=dropout_rate,
            inference=inference,
            key=key,
        )

        self.epsilon = jnp.asarray(epsilon)

        self.inference = inference
        self.reward_fn = reward_fn

    @overload
    def __call__(
        self,
        scene: TriangleScene,
        *,
        replay: Int[Array, " order"] | None = ...,
        replay_symettric: bool = ...,
        inference: Literal[True],
        key: PRNGKeyArray,
    ) -> Int[Array, " order"]: ...

    @overload
    def __call__(
        self,
        scene: TriangleScene,
        *,
        replay: Int[Array, " order"] | None = ...,
        replay_symettric: bool = ...,
        inference: Literal[False],
        key: PRNGKeyArray,
    ) -> tuple[Int[Array, " order"], Float[Array, ""], Float[Array, ""]]: ...

    def __call__(
        self,
        scene: TriangleScene,
        *,
        replay: Int[Array, " order"] | None = None,
        replay_symmetric: bool = False,
        inference: bool | None = None,
        key: PRNGKeyArray,
    ) -> (
        Int[Array, " order"]
        | tuple[Int[Array, " order"], Float[Array, ""], Float[Array, ""]]
    ):
        """
        Sample a path candidate in the given scene.

        Args:
            scene: The scene in which to sample the path candidate.
            replay: If provided, replay this path candidate instead of sampling it randomly.
            replay_symmetric: Whether to replay the symmetric path (swap transmitters and receivers).
            inference: Whether to run in inference mode (disables epsilon-greedy uniform sampling and dropout).
            key: PRNG key for randomness.

        Returns:
            If inference is True, return the sampled path candidate as an array of integers.
            If inference is False, return a tuple containing the sampled path candidate, the loss value, and the reward.

        """
        inference = self.inference if inference is None else inference

        xyz, tx, rx = unpack_scene(scene)

        if replay_symmetric and replay is not None:
            tx, rx = rx, tx
            replay = replay[::-1]

        # [num_objects 3 3]
        xyz = geometric_transformation(xyz, tx, rx)
        num_objects = xyz.shape[0]
        # [num_objects num_embeddings]
        objects_embeds = self.objects_encoder(xyz, active_objects=scene.mesh.mask)
        # [num_embeddings]
        scene_embeds = self.scene_encoder(
            objects_embeds,
            active_objects=scene.mesh.mask,
        )

        def scan_fn(
            carry: tuple[
                Int[Array, " order"],
                Float[Array, ""],
                Float[Array, "num_objects"],
                Int[Array, ""],
            ],
            x: tuple[Int[Array, ""], PRNGKeyArray],
        ) -> tuple[
            tuple[
                Int[Array, " order"],
                Float[Array, ""],
                Float[Array, "num_objects"],
                Int[Array, ""],
            ],
            Float[Array, ""],
        ]:
            partial_path_candidate, loss_value, edge_flows, previous_object = carry
            i, key = x

            next_object_key, next_edge_flows_key = jr.split(key)

            # Sample next object
            flow_policy = edge_flows
            if (
                not inference
            ):  # During training, we use uniform policy with probability epsilon
                epsilon_greedy_key, next_object_key = jr.split(next_object_key)
                uniform_policy = (
                    jnp.where(scene.mesh.mask, 1.0, 0.0)
                    if scene.mesh.mask is not None
                    else jnp.ones_like(flow_policy)
                )
                uniform_policy = uniform_policy.at[previous_object].set(
                    0.0,
                    wrap_negative_indices=False,
                )
                choose_uniform_policy = jr.bernoulli(
                    epsilon_greedy_key,
                    self.epsilon,
                ) | (edge_flows.sum() == 0.0)
                policy = jnp.where(
                    choose_uniform_policy,
                    uniform_policy,
                    flow_policy,
                )
            else:
                policy = flow_policy

            if self.action_masking:
                # Only implemented for second interaction
                # where second last object is TX
                is_second_interaction = i == 1
                previous_object_normal = normalize(
                    jnp.cross(xyz[previous_object, 0, :], xyz[previous_object, 1, :])
                )[0]

                previous_object_vertices = xyz[previous_object, :, :]
                in_vector = previous_object_vertices[
                    0, :
                ]  # From TX at (0,0,0) to previous object
                # [num_objects 3 3 3]
                out_vectors = (
                    previous_object_vertices[None, :, None, :] - xyz[:, None, :, :]
                )  # From all objects to previous object
                expected_dot_sign = jnp.sign(jnp.dot(in_vector, previous_object_normal))
                got_dot_sign = jnp.sign(jnp.dot(out_vectors, previous_object_normal))
                # The object is visible if at least one of its vertices is on the expected side
                visible = jnp.any(got_dot_sign == expected_dot_sign, axis=(1, 2))

                policy = jnp.where(
                    is_second_interaction,
                    jnp.where(visible, policy, 0.0),
                    policy,
                )

            if self.distance_based_weighting:
                centers = xyz.mean(axis=1)
                previous_object_center = jnp.where(
                    previous_object != -1,
                    centers[previous_object],
                    jnp.zeros(3),  # TX is at (0,0,0) after geometric transformation
                )
                d = jnp.linalg.norm(centers - previous_object_center, axis=-1)
                d += jnp.where(
                    i == self.order - 1,
                    jnp.linalg.norm(
                        centers
                        - jnp.array(
                            [0.0, 0.0, 1.0]
                        ),  # RX is at (0,0,1) after geometric transformation
                        axis=-1,
                    ),
                    0.0,
                )
                zero_d = d == 0.0
                d = jnp.where(zero_d, 1.0, d)
                w = 1 / (d * d)
                w = jnp.where((policy > 0) & (~zero_d), w, 0.0)
                w = jnp.where(w.sum() == 0.0, 1.0, w)
                w = w / w.sum()
                policy *= w

            if replay is not None:
                next_object = replay[i]
            else:
                next_object = jr.choice(next_object_key, num_objects, p=policy)

            # Update state variables
            partial_path_candidate = partial_path_candidate.at[i].set(next_object)
            state_embeds = self.state_encoder(
                partial_path_candidate,
                objects_embeds,
                active_objects=scene.mesh.mask,
            )

            # Compute loss (flow mismatch)
            parent_flow = edge_flows[next_object]

            reward = jnp.where(
                i == self.order - 1,
                self.reward_fn(partial_path_candidate, scene),
                0.0,
            )

            edge_flows = jnp.where(
                i == self.order - 1,
                jnp.zeros_like(edge_flows),
                self.flows(
                    objects_embeds,
                    scene_embeds,
                    state_embeds,
                    active_objects=scene.mesh.mask,
                    inference=inference,
                    key=next_edge_flows_key,
                )
                .at[next_object]
                .set(0.0),
            )

            loss_value += (parent_flow - edge_flows.sum() - reward) ** 2

            return (
                partial_path_candidate,
                loss_value,
                edge_flows,
                next_object,
            ), reward

        init_edge_flows_key, scan_key = jr.split(key)

        init_path_candidate = -jnp.ones(self.order, dtype=int)
        init_loss_value = jnp.array(0.0)
        init_state_embeds = jnp.zeros(self.state_encoder.out_size)
        init_edge_flows = self.flows(
            objects_embeds,
            scene_embeds,
            init_state_embeds,
            active_objects=scene.mesh.mask,
            inference=inference,
            key=init_edge_flows_key,
        )

        (path_candidate, loss_value, _, _), rewards = jax.lax.scan(
            scan_fn,
            (
                init_path_candidate,
                init_loss_value,
                init_edge_flows,
                jnp.array(-1),
            ),
            (jnp.arange(self.order), jr.split(scan_key, self.order)),
        )

        if inference:
            return path_candidate

        return path_candidate, loss_value, rewards[-1]
