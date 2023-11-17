# CBL-Conditional-Boundary-Loss
The official implementation of the accepted IEEE-TIP paper [Conditional Boundary Loss for Semantic Segmentation](https://ieeexplore.ieee.org/abstract/document/10173725).  
We provide the code of our CBL based on [our_maskformer](./maskformer/) and MMsegmentation.
The whole CBL MMsegmentation code base will be uploaded soon. For now, your can use our CBL in MMsegmentation following the instructions down below.

---
## Our Results  
More results of other models like Swin-B and PSP, together with the trained model weights file, will be updated soon once we finished the orgnization of our project.  
Temporary result table:  

### Cityscapes
| model       | Backbone    | iter    | Crop Size    | mIoU MMseg (single scale) | mIoU +CBL Ours (single scale)  |
| ----------- | ----------- | ------- |  ----------- | ------------------------- | ------------------------------ |    
| OCRNet      | HRNetW48    | 80K     | 512x1024     | 80.70                     | 81.95                          |
### ADE20K
| model             | Backbone  | mIoU(SS) | mIoU(MS)        |
| -----------       | --------- | -------- | --------        |
| MaskFormer        | Swin-B    | --       | 53.83(official) |
| MaskFormer +CBL   | Swin-B    | 53.49    | 54.89 [Trained Model Code:CBL0](https://pan.baidu.com/s/1vSP6DYBOs82O490RFQF1GQ?pwd=CBL0)      | 
| Mask2Former       | Swin-B    | --       | 55.07(official) |
| Mask2Former +CBL  | Swin-B    | 54.79    | 56.05 [Trained Model Code:CBL0](https://pan.baidu.com/s/1UaHXp_HjAiZ7wB7386a5nA?pwd=CBL0)      | 

## How to train MaskFormer +CBL model
We build our implementation based on the official code base of MaskFormer. Please refer to our MaskFormer code base at [our_maskformer](./maskformer/)
This implementation enables easy reproduction of our CBL on MaskFormer, **which do not need the above complicated steps for mmsegmentation**.
The trained MaskFormer+CBL model can also be found at [MaskFormer+CBL Trained Model Code:CBL0](https://pan.baidu.com/s/1vSP6DYBOs82O490RFQF1GQ?pwd=CBL0)
The trained Mask2Foremer+CBL model is also provided at [Mask2Former+CBL Trained Model Code:CBL0](https://pan.baidu.com/s/1UaHXp_HjAiZ7wB7386a5nA?pwd=CBL0)
## How to use our code in MMsegmentation
We follow the implementation of MMsegmentation. Here we provide the code of CBL based on the OCRHead in CBLocr_head.py.  
The class name of the OCRHead with our CBL is `New_ER5OCRHead`.   

1. Download our code and add the **CBLocr_head.py** to Path **mmseg/models/decode_head/ocr_head.py** in your MMsegmentation source code project.  
2. Import the ER5OCRHead class in the mmseg/models/decode_head/__init__.py:  
  change the line `from .ocr_head import OCRHead` as `from .ocr_head import OCRHead, New_ER5OCRHead`  
3. Add the code for generating Ground Truth boundary for training:  
  1. Download the **biou.py** and add it to mmseg/core/evaluation/biou.py  
  2. Import the functions in biou.py:  
        add the following line to the mmseg/core/evaluation/__init__.py   
        `from .biou import multi_class_gt_to_boundary`  
        then add `'multi_class_gt_to_boundary'`to the list of `__all__ = [xxx]`  
  3. Download the **boundary.py** and add it to mmseg/datasets/pipelines/boundary.py  
  4. Import the **GenerateBoundary** class in the mmseg/datasets/pipelines/__init__.py:  
        add the following line to the mmseg/datasets/pipelines/__init__.py  
        `from .boundary import GenerateBoundary`  
        then add `'GenerateBoundary'` to the list of `__all__ = [xxx]`  
4. Using our config.py to train a OCRNet.  
  For example, to train a OCRNet-HRNetW48 on cityscapes, please run the following code:  
  `sh tools/dist_train.sh YOUR_PATH_TO_THE_CONFIG/erocrnet_hr48_512x1024_80k_cityscapes_fp16.py 8`  

## TO-DO List(after accepted)
1.Upload the whole CBL project based on MMsegmentation (including CBL trained models with PSPNet, DeeplabV3+, Swin-B)  
2.Upload the whole CBL project based on MaskFormer (including CBL trained models with MaskFormer)  
3.Write a new instruction about how to run our CBL on the above-mentioned projects.  

