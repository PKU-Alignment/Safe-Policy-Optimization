#!/bin/bash
#Note: in this repo, mappo-L is named as happolag
TASK="ShadowHandOver"
ALGO="macpo"
NUM_ENVS=512

echo "Experiments started."
for seed in $(seq 0 2)
do
    python train.py --task=${TASK}  --seed $seed   --algo=${ALGO} --num_envs=${NUM_ENVS} --headless ${True}
done
echo "Experiments ended."
