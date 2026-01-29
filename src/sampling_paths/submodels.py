import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import (
    Array,
    Bool,
    Float,
    Int,
    PRNGKeyArray,
)


class ObjectsEncoder(eqx.Module):
    """Generate embeddings from triangle vertices."""

    out_size: int = eqx.field(static=True)

    mlp: eqx.nn.MLP

    def __init__(
        self,
        num_embeddings: int,
        width_size: int,
        depth: int,
        num_vertices_per_object: int = 3,
        *,
        key: PRNGKeyArray,
    ) -> None:
        """
        Initialize the objects encoder.

        Args:
            num_embeddings: Number of output embeddings per object.
            width_size: Width of the hidden layers in the MLP.
            depth: Number of hidden layers in the MLP.
            num_vertices_per_object: Number of vertices per object (default is 3 for triangles).
            key: The random key to be used.

        """
        self.out_size = num_embeddings

        self.mlp = eqx.nn.MLP(
            in_size=num_vertices_per_object * 3,
            out_size=num_embeddings,
            width_size=width_size,
            depth=depth,
            key=key,
        )

    def __call__(
        self,
        xyz: Float[Array, "num_objects num_vertices_per_object 3"],
        *,
        active_objects: Bool[Array, " num_objects"] | None = None,
        key: PRNGKeyArray | None = None,
    ) -> Float[Array, "num_objects num_embeddings"]:
        """
        Encode objects into embeddings.

        Args:
            xyz: The vertices of the objects to be encoded.
            active_objects: Boolean array indicating which objects are active.
            key: The random key to be used.
                Unused here.

        Returns:
            The embeddings for each object.

        """
        del key
        embeds = jax.vmap(self.mlp)(xyz.reshape(xyz.shape[0], -1))
        if active_objects is not None:
            return jnp.where(active_objects[:, None], embeds, 0.0)
        return embeds


class SceneEncoder(eqx.Module):
    """Generate scene embeddings from objects embeddings."""

    out_size: int = eqx.field(static=True)

    rho: eqx.nn.MLP

    def __init__(
        self,
        num_embeddings: int,
        width_size: int,
        depth: int,
        *,
        key: PRNGKeyArray,
    ) -> None:
        """
        Initialize the scene encoder.

        Args:
            num_embeddings: Number of output embeddings per object.
            width_size: Width of the hidden layers in the MLP.
            depth: Number of hidden layers in the MLP.
            key: The random key to be used.

        """
        self.out_size = num_embeddings
        self.rho = eqx.nn.MLP(
            in_size=num_embeddings,
            out_size=num_embeddings,
            width_size=width_size,
            depth=depth,
            key=key,
        )

    def __call__(
        self,
        objects_embeds: Float[Array, "num_objects num_embeddings"],
        *,
        active_objects: Bool[Array, " num_objects"] | None = None,
        key: PRNGKeyArray | None = None,
    ) -> Float[Array, " num_embeddings"]:
        """
        Encode a scene into embeddings.

        Args:
            objects_embeds: The embeddings of the objects in the scene.
            active_objects: Boolean array indicating which objects are active.
            key: The random key to be used.
                Unused here.

        Returns:
            The embeddings of the scene.

        """
        del key
        return self.rho(
            objects_embeds.mean(
                axis=0,
                where=active_objects[:, None] if active_objects is not None else None,
            ),
        )


class StateEncoder(eqx.Module):
    """Generate embeddings for a (partial) path candidate."""

    out_size: int = eqx.field(static=True)

    linear: eqx.nn.Linear

    def __init__(
        self,
        order: int,
        num_embeddings: int,
        *,
        key: PRNGKeyArray,
    ) -> None:
        """
        Initialize the state encoder.

        Args:
            order: The path order.
            num_embeddings: Number of output embeddings per state.
            key: The random key to be used.

        """
        self.out_size = order * num_embeddings

        self.linear = eqx.nn.Linear(
            in_features=self.out_size,
            out_features=self.out_size,
            key=key,
        )

    def __call__(
        self,
        partial_path_candidate: Int[Array, " order"],
        objects_embeds: Float[Array, "num_objects num_embeddings"],
        *,
        active_objects: Bool[Array, " num_objects"] | None = None,
        key: PRNGKeyArray | None = None,
    ) -> Float[Array, " out_size"]:
        """
        Encode a path candidate (state) into embeddings.

        Args:
            partial_path_candidate: The indices of the objects in the partial path candidate.
            objects_embeds: The embeddings of the objects in the scene.
            active_objects: Boolean array indicating which objects are active.
            key: The random key to be used.
                Unused here.

        Returns:
            The embeddings of the path candidate.

        """
        # N.B.: objects_embeds are already masked, so we do not need to use active_objects here
        del active_objects, key
        return self.linear(
            objects_embeds.at[partial_path_candidate]
            .get(mode="fill", wrap_negative_indices=False, fill_value=0.0)
            .reshape(self.out_size),
        )


class Flows(eqx.Module):
    """Compute unnormalized probabilities (flows) for each object."""

    mlp: eqx.nn.MLP
    dropout: eqx.nn.Dropout

    def __init__(
        self,
        *,
        in_size: int,
        width_size: int,
        depth: int,
        dropout_rate: float,
        inference: bool = False,
        key: PRNGKeyArray,
    ) -> None:
        """
        Initialize the flows module.

        Args:
            in_size: The input size.
            width_size: Width of the hidden layers in the MLP.
            depth: Number of hidden layers in the MLP.
            dropout_rate: Dropout rate to be used.
            inference: Whether to run in inference mode (disables dropout).
            key: The random key to be used.

        """
        self.mlp = eqx.nn.MLP(
            in_size=in_size,
            out_size="scalar",
            width_size=width_size,
            depth=depth,
            activation=jax.nn.leaky_relu,
            final_activation=jnp.exp,
            key=key,
        )
        self.dropout = eqx.nn.Dropout(dropout_rate, inference=inference)

    def __call__(
        self,
        objects_embeds: Float[Array, "num_objects num_embeddings"],
        scene_embeds: Float[Array, " num_scene_embeddings"],
        state_embeds: Float[Array, "  num_state_embeddings"],
        *,
        active_objects: Bool[Array, " num_objects"] | None = None,
        inference: bool | None = None,
        key: PRNGKeyArray | None = None,
    ) -> Float[Array, " num_objects"]:
        """
        Compute unnormalized probabilities (flows) for selecting the next object in the path given the current partial path candidate and the scene.

        Args:
            objects_embeds: Embeddings for all objects in the scene.
            scene_embeds: Embeddings for the scene.
            state_embeds: Embeddings for the current partial path candidate.
            active_objects: Boolean array indicating which objects are active.
            inference: Whether to run in inference mode (disables dropout).
            key: The random key to be used.
                Only required if not in inference mode.

        Returns:
            Unnormalized probabilities (flows) for each object.

        """
        flows = jax.vmap(
            lambda object_embeds, scene_embeds, state_embeds: self.mlp(
                jnp.concat((object_embeds, scene_embeds, state_embeds)),
            ),
            in_axes=(0, None, None),
        )(
            objects_embeds,
            scene_embeds,
            state_embeds,
        ).clip(min=1e-20)
        flows = self.dropout(flows, inference=inference, key=key)
        if active_objects is not None:
            flows = jnp.where(active_objects, flows, 0.0)
        return flows
