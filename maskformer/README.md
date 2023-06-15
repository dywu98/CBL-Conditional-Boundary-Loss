# MaskFormer: CBL implementation with MaskFormer

This is the official implementation of accepted IEEE TIP paper "Conditional Boundary Loss for Semantic Segmentation" on MaskFormer.

This CBL implementation is based on the official implementation [MaskFormer](https://alexander-kirillov.github.io/).



## Installation
Download this project, and install the requirements of [MaskFormer](https://alexander-kirillov.github.io/). 
For installing the requirements of [MaskFormer](https://alexander-kirillov.github.io/), please refer to its [installation instructions](INSTALL.md).

## Getting Started

See [Preparing Datasets for MaskFormer](datasets/README.md).

See [Getting Started with MaskFormer](GETTING_STARTED.md).

## Results
| model            | Backbone  | mIoU(SS) | mIoU(MS)        | Training Setting | Trained Model |
| -----------      | --------- | -------- | --------        | --------         | ------        |
| MaskFormer       | Swin-B    | --       | 53.83(official) |                  | [official model](https://dl.fbaipublicfiles.com/maskformer/semantic-ade20k/maskformer_swin_base_IN21k_384_bs16_160k_res640/model_final_45388b.pkl)      |
| MaskFormer +CBL  | Swin-B    | 53.49    | 54.89           | [config](configs\ade20k-150\swin\CBL.yaml) | [our model](https://pan.baidu.com/s/1vSP6DYBOs82O490RFQF1GQ?pwd=CBL0) code:CBL0|
