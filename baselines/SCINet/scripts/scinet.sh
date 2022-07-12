
#!/bin/bash


seeds="12 22 32 42 52"
gpu='0'


for seed in $seeds
do
  version="scinet_seed${seed}"
  custom_command="--seed ${seed}"
  CUDA_VISIBLE_DEVICES="${gpu}" python -u run_ETTh.py --data ETTh1 --features M  --seq_len 48 --label_len 24 --pred_len 24 --hidden-size 4 --stacks 1 --levels 3 --lr 3e-3 --batch_size 8 --dropout 0.5  ${custom_command} --model_name "${version}_etth1_M_I48_O24_lr3e-3_bs8_dp0.5_h4_s1l3"
  CUDA_VISIBLE_DEVICES="${gpu}" python -u run_ETTh.py --data ETTh1 --features M  --seq_len 96 --label_len 48 --pred_len 48 --hidden-size 4 --stacks 1 --levels 3 --lr 0.009 --batch_size 16 --dropout 0.25  ${custom_command} --model_name "${version}_etth1_M_I96_O48_lr0.009_bs16_dp0.25_h4_s1l3"
  CUDA_VISIBLE_DEVICES="${gpu}" python -u run_ETTh.py --data ETTh1 --features M  --seq_len 336 --label_len 168 --pred_len 168 --hidden-size 4 --stacks 1 --levels 3 --lr 5e-4 --batch_size 32 --dropout 0.5  ${custom_command} --model_name "${version}_etth1_M_I336_O168_lr5e-4_bs32_dp0.5_h4_s1l3"
  CUDA_VISIBLE_DEVICES="${gpu}" python -u run_ETTh.py --data ETTh1 --features M  --seq_len 336 --label_len 336 --pred_len 336 --hidden-size 1 --stacks 1 --levels 4 --lr 1e-4 --batch_size 512 --dropout 0.5  ${custom_command} --model_name "${version}_etth1_M_I336_O336_lr1e-4_bs512_dp0.5_h1_s1l4"
  CUDA_VISIBLE_DEVICES="${gpu}" python -u run_ETTh.py --data ETTh1 --features M  --seq_len 736 --label_len 720 --pred_len 720 --hidden-size 1 --stacks 1 --levels 5 --lr 5e-5 --batch_size 256 --dropout 0.5  ${custom_command} --model_name "${version}_etth1_M_I736_O720_lr5e-5_bs256_dp0.5_h1_s1l5"
  CUDA_VISIBLE_DEVICES="${gpu}" python -u run_ETTh.py --data ETTh1 --features M  --seq_len 480 --label_len 480 --pred_len 960 --hidden-size 1 --stacks 1 --levels 4 --lr 1e-4 --batch_size 512 --dropout 0.5  ${custom_command} --model_name "${version}_etth1_M_I480_O960_lr1e-4_bs512_dp0.5_h1_s1l4"

  CUDA_VISIBLE_DEVICES="${gpu}" python -u run_ETTh.py --data ETTh2 --features M  --seq_len 48 --label_len 24 --pred_len 24 --hidden-size 8 --stacks 1 --levels 3 --lr 0.007 --batch_size 16 --dropout 0.25  ${custom_command} --model_name "${version}_etth2_M_I48_O24_lr7e-3_bs16_dp0.25_h8_s1l3"
  CUDA_VISIBLE_DEVICES="${gpu}" python -u run_ETTh.py --data ETTh2 --features M  --seq_len 96 --label_len 48 --pred_len 48 --hidden-size 4 --stacks 1 --levels 4 --lr 0.007 --batch_size 4 --dropout 0.5  ${custom_command} --model_name "${version}_etth2_M_I96_O48_lr7e-3_bs4_dp0.5_h4_s1l4"
  CUDA_VISIBLE_DEVICES="${gpu}" python -u run_ETTh.py --data ETTh2 --features M  --seq_len 336 --label_len 168 --pred_len 168 --hidden-size 0.5 --stacks 1 --levels 4 --lr 5e-5 --batch_size 16 --dropout 0.5  ${custom_command} --model_name "${version}_etth2_M_I336_O168_lr5e-5_bs16_dp0.5_h0.5_s1l4"
  CUDA_VISIBLE_DEVICES="${gpu}" python -u run_ETTh.py --data ETTh2 --features M  --seq_len 336 --label_len 336 --pred_len 336 --hidden-size 1 --stacks 1 --levels 4 --lr 5e-5 --batch_size 128 --dropout 0.5  ${custom_command} --model_name "${version}_eetth2_M_I336_O336_lr5e-5_bs128_dp0.5_h1_s1l4"
  CUDA_VISIBLE_DEVICES="${gpu}" python -u run_ETTh.py --data ETTh2 --features M  --seq_len 736 --label_len 720 --pred_len 720 --hidden-size 4 --stacks 1 --levels 5 --lr 1e-5 --batch_size 32 --dropout 0.5  ${custom_command} --model_name "${version}_etth2_M_I736_O720_lr1e-5_bs32_dp0.5_h4_s1l5"
  CUDA_VISIBLE_DEVICES="${gpu}" python -u run_ETTh.py --data ETTh2 --features M  --seq_len 480 --label_len 480 --pred_len 960 --hidden-size 1 --stacks 1 --levels 4 --lr 5e-5 --batch_size 128 --dropout 0.5  ${custom_command} --model_name "${version}_etth2_M_I480_O960_lr5e-5_bs128_dp0.5_h1_s1l4"

  CUDA_VISIBLE_DEVICES="${gpu}" python -u run_ETTh.py --data ETTm1 --features M  --seq_len 48 --label_len 24 --pred_len 24 --hidden-size 4 --stacks 1 --levels 3 --lr 0.005 --batch_size 32 --dropout 0.5  ${custom_command} --model_name "${version}_ettm1_M_I48_O24_lr7e-3_bs16_dp0.25_h8_s1l3"
  CUDA_VISIBLE_DEVICES="${gpu}" python -u run_ETTh.py --data ETTm1 --features M  --seq_len 96 --label_len 48 --pred_len 48 --hidden-size 4 --stacks 2 --levels 4 --lr 0.001 --batch_size 16 --dropout 0.5  ${custom_command} --model_name "${version}_ettm1_M_I96_O48_lr1e-3_bs16_dp0.5_h4_s2l4"
  CUDA_VISIBLE_DEVICES="${gpu}" python -u run_ETTh.py --data ETTm1 --features M  --seq_len 384 --label_len 96 --pred_len 96 --hidden-size 0.5 --stacks 2 --levels 4 --lr 5e-5 --batch_size 32 --dropout 0.5  ${custom_command} --model_name "${version}_ettm1_M_I384_O96_lr5e-5_bs32_dp0.5_h0.5_s2l4"
  CUDA_VISIBLE_DEVICES="${gpu}" python -u run_ETTh.py --data ETTm1 --features M  --seq_len 672 --label_len 288 --pred_len 288 --hidden-size 4 --stacks 1 --levels 5 --lr 1e-5 --batch_size 32 --dropout 0.5  ${custom_command} --model_name "${version}_ettm1_M_I672_O288_lr1e-5_bs32_dp0.5_h0.5_s1l5"
  CUDA_VISIBLE_DEVICES="${gpu}" python -u run_ETTh.py --data ETTm1 --features M  --seq_len 672 --label_len 672 --pred_len 672 --hidden-size 4 --stacks 2 --levels 5 --lr 1e-5 --batch_size 32 --dropout 0.5  ${custom_command} --model_name "${version}_ettm1_M_I672_O672_lr1e-5_bs32_dp0.5_h4_s2l5"
  CUDA_VISIBLE_DEVICES="${gpu}" python -u run_ETTh.py --data ETTm1 --features M  --seq_len 672 --label_len 672 --pred_len 1344 --hidden-size 4 --stacks 2 --levels 5 --lr 1e-5 --batch_size 32 --dropout 0.5  ${custom_command} --model_name "${version}_ettm1_M_I672_O1344_lr1e-5_bs32_dp0.5_h4_s2l5"

  CUDA_VISIBLE_DEVICES="${gpu}" python -u run_ETTh.py --data ECL --features M  --seq_len 736 --label_len 720 --pred_len 720 --hidden-size 1 --stacks 1 --levels 3 --lr 5e-5 --batch_size 16 --dropout 0.5  ${custom_command} --model_name "${version}_ECL_M_I736_O720_lr1e-5_bs32_dp0.5_h4_s1l5"
  CUDA_VISIBLE_DEVICES="${gpu}" python -u run_ETTh.py --data ECL --features M  --seq_len 480 --label_len 480 --pred_len 960 --hidden-size 1 --stacks 1 --levels 3 --lr 5e-5 --batch_size 16 --dropout 0.5  ${custom_command} --model_name "${version}_ECL_M_I480_O960_lr1e-5_bs32_dp0.5_h4_s1l5"
  CUDA_VISIBLE_DEVICES="${gpu}" python -u run_ETTh.py --data ECL --features M  --seq_len 48 --label_len 24 --pred_len 24 --hidden-size 1 --stacks 1 --levels 3 --lr 5e-5 --batch_size 16 --dropout 0.5  ${custom_command} --model_name "${version}_ECL_M_I48_O24_lr7e-3_bs16_dp0.25_h8_s1l3"
  CUDA_VISIBLE_DEVICES="${gpu}" python -u run_ETTh.py --data ECL --features M  --seq_len 96 --label_len 48 --pred_len 48 --hidden-size 1 --stacks 1 --levels 3 --lr 5e-5 --batch_size 16 --dropout 0.5  ${custom_command} --model_name "${version}_ECL_M_I96_O48_lr7e-3_bs4_dp0.5_h4_s1l4"
  CUDA_VISIBLE_DEVICES="${gpu}" python -u run_ETTh.py --data ECL --features M  --seq_len 336 --label_len 168 --pred_len 168 --hidden-size 1 --stacks 1 --levels 3 --lr 5e-5 --batch_size 16 --dropout 0.5  ${custom_command} --model_name "${version}_ECL_M_I336_O168_lr5e-5_bs16_dp0.5_h0.5_s1l4"
  CUDA_VISIBLE_DEVICES="${gpu}" python -u run_ETTh.py --data ECL --features M  --seq_len 336 --label_len 336 --pred_len 336 --hidden-size 1 --stacks 1 --levels 3 --lr 5e-5 --batch_size 16 --dropout 0.5  ${custom_command} --model_name "${version}_ECL_M_I336_O336_lr5e-5_bs128_dp0.5_h1_s1l4"

done