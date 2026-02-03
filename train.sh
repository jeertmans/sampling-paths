for order in 1 2 3
do
for action_masking in "--action-masking" "--no-action-masking"
do
for distance_based_weighting in "--distance-based-weighting" "--no-distance-based-weighting"
do
echo $order $action_masking $distance_based_weighting
uv run train-path-sampler $action_masking $distance_based_weighting --order $order || true
done
done
done
done