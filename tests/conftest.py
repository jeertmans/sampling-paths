import chex
import jax
import jax.numpy as jnp
import optax
import pytest
from differt.geometry import TriangleMesh
from differt.scene import TriangleScene
from jaxtyping import Array, Int, PRNGKeyArray

from sampling_paths.agent import Agent
from sampling_paths.model import Model
from sampling_paths.submodels import Flows, ObjectsEncoder, SceneEncoder, StateEncoder


@pytest.fixture
def seed() -> int:
    return 1234


@pytest.fixture
def key(seed: int) -> PRNGKeyArray:
    return jax.random.key(seed)


@pytest.fixture(params=[1, 2, 3])
def order(request: pytest.FixtureRequest) -> int:
    return request.param


@pytest.fixture
def num_embeddings() -> int:
    return 32


@pytest.fixture
def width_size(num_embeddings: int) -> int:
    return 2 * num_embeddings


@pytest.fixture
def depth() -> int:
    return 2


@pytest.fixture
def dropout_rate() -> float:
    return 0.0


@pytest.fixture
def epsilon() -> float:
    return 0.1


@pytest.fixture
def model(
    order: int,
    num_embeddings: int,
    width_size: int,
    depth: int,
    dropout_rate: float,
    epsilon: float,
    key: PRNGKeyArray,
) -> Model:
    return Model(
        order=order,
        num_embeddings=num_embeddings,
        width_size=width_size,
        depth=depth,
        dropout_rate=dropout_rate,
        epsilon=epsilon,
        key=key,
    )


@pytest.fixture
def flows(model: Model) -> Flows:
    return model.flows


@pytest.fixture
def objects_encoder(model: Model) -> ObjectsEncoder:
    return model.objects_encoder


@pytest.fixture
def scene_encoder(model: Model) -> SceneEncoder:
    return model.scene_encoder


@pytest.fixture
def state_encoder(model: Model) -> StateEncoder:
    return model.state_encoder


@pytest.fixture
def batch_size() -> int:
    return 64


@pytest.fixture
def optim() -> optax.GradientTransformationExtraArgs:
    return optax.adam(3e-4)


@pytest.fixture
def delta_epsilon() -> float:
    return 0.0


@pytest.fixture
def min_epsilon(epsilon: float) -> float:
    return epsilon


@pytest.fixture
def agent(
    model: Model,
    batch_size: int,
    optim: optax.GradientTransformationExtraArgs,
    delta_epsilon: float,
    min_epsilon: float,
) -> Agent:
    return Agent(
        model=model,
        batch_size=batch_size,
        optim=optim,
        delta_epsilon=delta_epsilon,
        min_epsilon=min_epsilon,
    )


@pytest.fixture(params=[True, False])
def inference(request: pytest.FixtureRequest) -> bool:
    return request.param


@pytest.fixture
def scene(key: PRNGKeyArray) -> TriangleScene:
    r = 1.0 / jnp.sqrt(3)
    vertices = jnp.array(
        [
            [0.0, r, 0.0],  # Top vertex
            [-0.5, -0.5 * r, 0.0],  # Bottom left
            [0.5, -0.5 * r, 0.0],  # Bottom right
        ]
    )
    triangles = jnp.array(
        [
            [0, 1, 2],
        ]
    )

    triangle = TriangleMesh(vertices=vertices, triangles=triangles)

    mesh = sum(
        (triangle.translate(jnp.array([0, 0, dz])) for dz in [-2, -1, 1, 2]),
        start=TriangleMesh.empty(),
    )

    # We add inactive objects to the scene
    # to make sure the agent effectively ignores them
    mesh += mesh.sample(0, by_masking=True, key=key)

    assert mesh.mask is not None
    assert mesh.num_triangles == 8

    return TriangleScene(
        transmitters=jnp.array([0, 0, +0.5]),
        receivers=jnp.array([0, 0, -0.5]),
        mesh=mesh,
    )


@pytest.fixture
def masked_actions(
    scene: TriangleScene,
) -> dict[int, Int[Array, " num_masked_actions"]]:
    # List unreachable objects in the scene after a first reflection on each object
    tx = scene.transmitters.reshape(3)
    # [num_objects 3 3]
    xyz = scene.mesh.triangle_vertices
    # [num_objects 3]
    normals = scene.mesh.normals

    masked_objects = {}
    for i in range(xyz.shape[0]):
        if scene.mesh.mask is not None and not scene.mesh.mask[i]:
            # Inactive object, pass
            masked_objects[i] = jnp.array([], dtype=int)
            continue

        # [3]
        obj_normal = normals[i]
        # [3 3]
        obj_vertices = xyz[i]
        # [3 3]
        i_vecs = tx - obj_vertices
        # [num_objects 3 3 3]
        r_vecs = obj_vertices[None, None, :, :] - xyz[:, :, None, :]
        # [3]
        i_dot = jnp.sum(i_vecs * obj_normal, axis=-1)
        # [num_objects 3 3]
        r_dot = jnp.sum(r_vecs * obj_normal, axis=-1)
        got_dot_sign = jnp.sign(r_dot)
        expected_dot_sign = jnp.sign(i_dot)
        reachable = jnp.any(
            got_dot_sign[..., None] != expected_dot_sign, axis=(1, 2, 3)
        )
        if scene.mesh.mask is not None:
            reachable = jnp.where(~scene.mesh.mask, True, reachable)
        masked_objects[i] = jnp.argwhere(~reachable).flatten()

    chex.assert_trees_all_equal(
        masked_objects,
        {
            0: jnp.array(
                [], dtype=int
            ),  # All objects are reachable after reflecting on object 0
            1: jnp.array([0]),  # Object 0 is unreachable after reflecting on object 1
            2: jnp.array([3]),  # Object 3 is unreachable after reflecting on object 2
            3: jnp.array(
                [], dtype=int
            ),  # All objects are reachable after reflecting on object 3
            4: jnp.array([], dtype=int),  # Inactive object
            5: jnp.array([], dtype=int),  # Inactive object
            6: jnp.array([], dtype=int),  # Inactive object
            7: jnp.array([], dtype=int),  # Inactive object
        },
    )

    return masked_objects
