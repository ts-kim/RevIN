
#!/bin/bash


seeds="12"
gpu='0'


for seed in $seeds
do
  version="scinet_ours_seed${seed}"
  custom_command="--seed ${seed} --ours --evaluate"
  CUDA_VISIBLE_DEVICES="${gpu}" python -u run_ETTh.py --data ETTh2 --features M  --seq_len 480 --label_len 480 --pred_len 960 --hidden-size 1 --stacks 1 --levels 4 --lr 5e-5 --batch_size 128 --dropout 0.5  ${custom_command} --model_name "${version}_etth2_M_I480_O960_lr5e-5_bs128_dp0.5_h1_s1l4"
done