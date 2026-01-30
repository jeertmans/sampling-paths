for sampling_objects in "--sample-objects" "--no-sample-objects"
do
for sampling_in_canyon in "--sample-in-canyon" "--no-sample-in-canyon"
do
for including_floor in "--include-floor" "--no-include-floor"
do
for order in 2 3
do
echo $sampling_objects $sampling_in_canyon $including_floor $order
uv run train-path-sampler $sampling_objects $sampling_in_canyon $including_floor --order $order || true
done
done
done
done