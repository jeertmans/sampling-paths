for epsilon in 0.1 0.2 0.5
do
for alpha in 0.25 0.5 0.75
do
for replay_symmetric in "--no-replay-symmetric" "--replay-symmetric"
do
echo $epsilon $alpha $replay_symmetric
uv run train-path-sampler $replay_symmetric --epsilon $epsilon --alpha $alpha --num-episodes 300000 --num-embeddings 64 --order 2 || true
uv run train-path-sampler $replay_symmetric --epsilon $epsilon --alpha $alpha --num-episodes 300000 --num-embeddings 64 --order 3 || true
done
done
done
# IMPORTANT: last iteration failed
