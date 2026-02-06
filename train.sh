for order in 3 2 1
do
for action_masking in "--action-masking" "--no-action-masking"
do
for distance_based_weighting in "--distance-based-weighting" "--no-distance-based-weighting"
do
echo $order $action_masking $distance_based_weighting
uv run train-path-sampler $action_masking $distance_based_weighting --order $order --results-dir action_masking_and_distance_based_weighting || true
done
done
done

for order in 3 2 1
do
for replay_buffer in "--replay-buffer" "--no-replay-buffer"
do
for exploratory_policy in "--exploratory-policy" "--no-exploratory-policy"
do
echo $order $replay_buffer $exploratory_policy
uv run train-path-sampler $replay_buffer $exploratory_policy --order $order --results-dir replay_buffer_and_exploratory_policy || true
done
done
done

for order in 3 2 1
do
for replay_symmetric in "--replay-symmetric" "--no-replay-symmetric"
do
echo $order $replay_symmetric
uv run train-path-sampler $replay_symmetric --order $order --results-dir replay_symmetric || true
done
done
