
for sample_objects in "--no-sample-objects" "--sample-objects"
do
for order in 1 2 3
do
for sample_in_canyon in "--no-sample-in-canyon" "--sample-in-canyon"
do
for include_floor in "--no-include-floor" "--include-floor"
do
echo $order $sample_objects $sample_in_canyon $include_floor
uv run train-path-sampler $sample_objects $sample_in_canyon $include_floor --order $order --results-dir sample_objects_and_sample_in_canyon_and_include_floor || true
done
done
done
done
