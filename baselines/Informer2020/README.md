# RevIN + Informer

This is an example code of RevIN.
The code is based on [Informer](https://github.com/zhouhaoyi/Informer2020).

## Usage

- To obtain the data for training, please follow the instructions at the github repository of [Informer](https://github.com/zhouhaoyi/Informer2020).

- To activate Reversible Instance Normalization, add `--use_RevIN` to your command.


#### Examples

```
# with RevIN
python -u main_informer.py --model informer --data ETTh2 --features M --seq_len 720 --label_len 336 --pred_len 720 --e_layers 3 --d_layers 2 --attn prob --des 'Exp' --itr 5 --use_RevIN

# without RevIN
python -u main_informer.py --model informer --data ETTh2 --features M --seq_len 720 --label_len 336 --pred_len 720 --e_layers 3 --d_layers 2 --attn prob --des 'Exp' --itr 5
```


## Citation

If you find this work or code is helpful in your research, please cite:
```
@inproceedings{kim2021reversible,
  title     = {Reversible Instance Normalization for Accurate Time-Series Forecasting against Distribution Shift},
  author    = {Kim, Taesung and 
               Kim, Jinhee and 
               Tae, Yunwon and 
               Park, Cheonbok and 
               Choi, Jang-Ho and 
               Choo, Jaegul},
  booktitle = {International Conference on Learning Representations},
  year      = {2021},
  url       = {https://openreview.net/forum?id=cGDAkQo1C0p}
}
```
