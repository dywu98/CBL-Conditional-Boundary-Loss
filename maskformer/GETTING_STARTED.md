## Getting Started with MaskFormer+CBL

This document provides a brief intro of the usage of MaskFormer+CBL.

Please see [Getting Started with Detectron2](https://github.com/facebookresearch/detectron2/blob/master/GETTING_STARTED.md) for full usage.


### Training & Evaluation in Command Line

For training and inference the original MaskFormer, please run `train_net.py`.

**To train our MaskFormer+CBL model, please run `train_net_biou.py`**

To train a model with "train_net.py" or "train_net_biou.py", first
setup the corresponding datasets following
[datasets/README.md](./datasets/README.md), which is the same with that of the official MaskFormer repo, 
then run:
```
./train_net_biou.py --num-gpus 8 \
  --config-file configs\ade20k-150\swin\CBL.yaml
```

To evaluate a model's performance, use
```
./train_net_biou.py \
  --config-file configs\ade20k-150\swin\CBL.yaml \
  --eval-only MODEL.WEIGHTS /path/to/checkpoint_file
```
For more options, see `./train_net_biou.py -h`.

To evaluate the MS+FLIP results, please follow this setting in the config file:
```
TEST:
  AUG:
    ENABLED: true
    FLIP: true
```
During training, you can just set these as False to save validation time.
