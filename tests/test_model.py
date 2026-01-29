import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest
from differt.geometry import (
    rotation_matrix_along_z_axis,
)
from differt.scene import TriangleScene
from jaxtyping import Array, Int, PRNGKeyArray

from sampling_paths.model import Model


class TestModel:
    def test_model(self, model: Model, scene: TriangleScene, key: PRNGKeyArray) -> None:
        mask = scene.mesh.mask
        if mask is None:
            mask = jnp.array([], dtype=bool)
        inactive_objects = jnp.argwhere(~mask)
        path_candidates = jax.vmap(lambda key: model(scene, inference=True, key=key))(
            jr.split(key, 1_000)
        )
        for path_candidate in path_candidates:
            # model should never generate a path that contains the same object twice in a row
            assert not (path_candidate[:-1] == path_candidate[1:]).any(), (
                f"Path candidate should not contain the same object twice in a row, got: {path_candidate}"
            )
            # model should never generate a path that contains inactive objects
            assert not jnp.isin(inactive_objects, path_candidate).any(), (
                f"Path candidate should not contain inactive objects, got: {path_candidate}, but inactive objects are: {inactive_objects}"
            )

    @pytest.mark.parametrize("inference", [True, False])
    def test_action_masking(
        self,
        inference: bool,
        model: Model,
        scene: TriangleScene,
        masked_actions: dict[int, Int[Array, " num_masked_actions"]],
        key: PRNGKeyArray,
    ) -> None:
        if model.order < 2:
            pytest.skip("Action masking only applies for order >= 2")
        if not model.action_masking:
            pytest.xfail(
                "Model should have action masking enabled for this test to pass"
            )

        def transform_scene_and_sample_path_candidates(
            key: PRNGKeyArray,
        ) -> Int[Array, " order"]:
            key_rotate, key_sample_path_candidate = jr.split(key, 2)
            rot_z = jr.uniform(key_rotate, minval=-jnp.pi, maxval=jnp.pi)
            transformed_scene = eqx.tree_at(
                lambda s: s.mesh,
                scene,
                scene.mesh.rotate(rotation_matrix_along_z_axis(rot_z)),
            )
            if inference:
                return model(
                    transformed_scene, inference=True, key=key_sample_path_candidate
                )
            return model(
                transformed_scene, inference=False, key=key_sample_path_candidate
            )[0]

        path_candidates = jax.vmap(transform_scene_and_sample_path_candidates)(
            jr.split(key, 1_000)
        )
        for path_candidate in path_candidates:
            first_object = path_candidate[0]
            second_object = path_candidate[1]
            # model should never sample an object that is masked from the first object reflection
            assert not jnp.isin(second_object, masked_actions[int(first_object)]), (
                f"Object {second_object} is masked after reflecting on object {first_object}, "
                f"but got path candidate: {path_candidate}"
            )
