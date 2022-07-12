# RevIN + SCINet
Pytorch implementation of "Reversible Instance Normalization for Accurate Time-Series Forecasting against Distribution Shift"

This code is based on [SCINet](https://github.com/cure-lab/SCINet).

## Prerequisites
Install following dependencies:
- numpy
- pytorch
- pandas
- scikit_learn
- tensorboard

## Datasets
The ETT datasets can be obtained from https://github.com/zhouhaoyi/ETDataset.
The ECL dataset, preprocessed to have an hourly basis, can be obtained from https://github.com/zhouhaoyi/Informer2020.
The datasets should be located in `data/` folder.

## Pretrained models
The pretrained models `checkpoint.pth` for the ETTh2 dataset with prediction length of 960 are provided in the `exp/ETT_checkpoints/` folder.
To evaluate the performance of pretrained model, run the `scripts/scinet_with_RevIN_test.sh` file.
```
bash scripts/scinet_with_RevIN_test.sh
```

## Final hyperparameter settings
All of the hyperparameter settings for the model are available at `scripts/scinet_with_RevIN.sh` and `run_ETTh.py` files.

## Training
To activate Reversible Instance Normalization, add `--ours` to your command.

To train and evaluate SCINet with RevIN, run the `scripts/scinet_with_RevIN.sh` file.
```
bash scripts/scinet_with_RevIN.sh
```

To train and evaluate SCINet (baseline model), run the `scripts/scinet.sh` file.
```
bash scripts/scinet.sh
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
