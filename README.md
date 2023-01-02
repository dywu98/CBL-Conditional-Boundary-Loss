# CBL-Conditional-Boundary-Loss
The official implementation of IEEE-TIP paper under review.
Please note that right now this project is only the simple one **only including the core code** of our CBL implementation.
The whole implementations that are easy to use based on MMsegmentation will be released soon after the accpectance of our paper. 
However, you can still use our CBL in MMsegmentation following our instructions.
Thus, it may seems not convinient to reproduce our method right now. But, you will find it's easy to use CBL and reproduce the results in our paper once the whole MMsegmentation project is uploaded.

---

## How to use our code in MMsegmentation
We follow the implementation of MMsegmentation. Here we provide the code of CBL based on the OCRHead in CBLocr_head.py.
The class name of the OCRHead with our CBL is `New_ER5OCRHead`. 
1.Download our code and add the **CBLocr_head.py** to Path **mmseg/models/decode_head/ocr_head.py** in your MMsegmentation source code project.
2.Import the ER5OCRHead class in the mmseg/models/decode_head/__init__.py:
  change the line `from .ocr_head import OCRHead` as `from .ocr_head import OCRHead, New_ER5OCRHead`
3.Add the code for generating Ground Truth boundary for training:
  3.1. Download the **biou.py** and add it to mmseg/core/evaluation/biou.py
  3.2. Import the functions in biou.py:
        add the following line to the mmseg/core/evaluation/__init__.py 
        `from .biou import multi_class_gt_to_boundary`
        then add `'multi_class_gt_to_boundary'`to the list of `__all__ = [xxx]`
  3.3. Download the **boundary.py** and add it to mmseg/datasets/pipelines/boundary.py
  3.4. Import the **GenerateBoundary** class in the mmseg/datasets/pipelines/__init__.py:
        add the following line to the mmseg/datasets/pipelines/__init__.py
        `from .boundary import GenerateBoundary`
        then add `'GenerateBoundary'` to the list of `__all__ = [xxx]`
4.Using our config.py to train a OCRNet.
  For example, to train a OCRNet-HRNetW48 on cityscapes, please run the following code:
  `sh tools/dist_train.sh YOUR_PATH_TO_THE_CONFIG/erocrnet_hr48_512x1024_80k_cityscapes_fp16.py 8`

##TO-DO List(after accepted)
1.Upload the whole CBL project based on MMsegmentation (including CBL trained models with PSPNet, DeeplabV3+, Swin-B)
2.Upload the whole CBL project based on MaskFormer (including CBL trained models with MaskFormer)
3.Write a new instruction about how to run our CBL on the above-mentioned projects.

