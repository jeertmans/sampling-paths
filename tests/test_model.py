import jax.numpy as jnp
import jax.random as jr
import pytest
from differt.scene import TriangleScene
from jaxtyping import Array, Int, PRNGKeyArray

from sampling_paths.model import Model


class TestModel:
    def test_model(self, model: Model, scene: TriangleScene, key: PRNGKeyArray) -> None:
        mask = scene.mesh.mask
        if mask is None:
            mask = jnp.array([], dtype=bool)
        inactive_objects = jnp.argwhere(~mask)
        for sample_key in jr.split(key, 100):
            path_candidate = model(scene, inference=True, key=sample_key)
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
        assert model.action_masking, (
            "Model should have action masking enabled for this test"
        )
        if model.order < 2:
            pytest.skip("Action masking only applies for order >= 2")
        for sample_key in jr.split(key, 100):
            if inference:
                path_candidate = model(scene, inference=True, key=sample_key)
            else:
                path_candidate, _, _ = model(scene, inference=False, key=sample_key)
            first_object = path_candidate[0]
            second_object = path_candidate[1]
            # model should never sample a object that is masked from the first object reflection
            assert not jnp.isin(second_object, masked_actions[int(first_object)]), (
                f"Object {second_object} is masked after reflecting on object {first_object}, "
                f"but got path candidate: {path_candidate}"
            )
