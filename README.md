# RevIN (ICLR 2022) - Official PyTorch Implementation

[<ins>__[Paper]__</ins>](https://openreview.net/pdf?id=cGDAkQo1C0p) &nbsp; 
&nbsp; 
 [<ins>__[Project page]__</ins>](https://seharanul17.github.io/RevIN/)

## Introduction

This is the official PyTorch implementation of [Reversible Instance Normalization for Accurate Time-Series Forecasting against Distribution Shift](https://openreview.net/forum?id=cGDAkQo1C0p).

Statistical properties such as mean and variance often change over time in time series, i.e., time-series data suffer from a distribution shift problem. This change in temporal distribution is one of the main challenges that prevent accurate time-series forecasting. To address this issue, we propose a simple yet effective normalization method called reversible instance normalization (RevIN), a generally-applicable normalization-and-denormalization method with learnable affine transformation. The proposed method is symmetrically structured to remove and restore the statistical information of a time-series instance, leading to significant performance improvements in time-series forecasting, as shown in Fig. 1. We demonstrate the effectiveness of RevIN via extensive quantitative and qualitative analyses on various real-world datasets, addressing the distribution shift problem.


## Environment

The code was developed using python 3.8 on Ubuntu 18.04. 

The experiments were performed on a single NVIDIA TITAN RTX or NVIDIA TITAN Xp.


## Quick start

### Installation
1. Install PyTorch >= v.1.8.0 following the [official instruction](https://pytorch.org/). 
   - tested with PyTorch v.1.8.0 and PyTorch v.1.11.0.
2. Clone this repository:
    ```
    git clone https://github.com/ts-kim/RevIN.git
    ```

### Usage
RevIN calculates the mean and standard deviation of each feature separately for each sequence in a mini-batch.


To be reversible, the input and output tensors should have the same number of features.
The input tensors should be provided as *(..., feature)*.
For example,
- x_in: (batch_size, sequence_length, num_features)
- x_out: (batch_size, prediction_length, num_features)

RevIN can be added in any arbitrarily chosen layers of a model as follows:
```
>>> from RevIN import RevIN
>>> revin_layer = RevIN(num_features)
>>> x_in = revin_layer(x_in, 'norm')
>>> x_out = blocks(x_in) # your model or subnetwork within the model
>>> x_out = revin_layer(x_out, 'denorm')
```

### Baselines
We updated the training and evaluation codes for the baselines, [Informer](https://github.com/zhouhaoyi/Informer2020) and [SCINet](https://github.com/cure-lab/SCINet).


Please go to the [`baselines`](https://github.com/ts-kim/RevIN/tree/master/baselines) forder or the link below.
- [`Informer`](https://github.com/ts-kim/RevIN/tree/master/baselines/Informer2020)
- [`SCINet`](https://github.com/ts-kim/RevIN/tree/master/baselines/SCINet)

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
