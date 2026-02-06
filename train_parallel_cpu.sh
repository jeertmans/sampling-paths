#!/bin/bash

FAIL=0

export JAX_PLATFORMs=cpu
export JAX_PLATFORM_NAME=cpu

echo "starting"

for order in 3 2 1
do
for sample_objects in "--sample-objects" "--no-sample-objects"
do
for sample_in_canyon in "--sample-in-canyon" "--no-sample-in-canyon"
do
for include_floor in "--include-floor" "--no-include-floor"
do
echo $order $sample_objects $sample_in_canyon $include_floor
uv run train-path-sampler --depth $depth $sample_objects $sample_in_canyon $include_floor --order $order --results-dir sample_objects_and_sample_in_canyon_and_include_floor &
done
done
done
done

for job in `jobs -p`
do
echo $job
    wait $job || let "FAIL+=1"
done

echo $FAIL

if [ "$FAIL" == "0" ];
then
echo "YAY!"
else
echo "FAIL! ($FAIL)"
fi