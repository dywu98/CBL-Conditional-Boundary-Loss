# This is the code to generate boundary given gt and calculate BIoU.
# Please move this file to the mmseg/core/evaluation/biou.py in your mmsegmentation source project
# Make sure that the functions are imported in the mmseg/core/evaluation/__init__.py

import os
import cv2
import argparse
import numpy as np
from torch._C import dtype
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.multiprocessing as mp

# from config import config
# from network import PSPNet
# #from utils.crf import DenseCRF

# from utils.pyt_utils import ensure_dir, link_file, load_model, \
#     parse_devices
# from utils.visualize import print_iou, show_img
# from engine.evaluator import Evaluator
# from engine.logger import get_logger
# from seg_opr.metric import hist_info, compute_score
# from datasets import vivo

# logger = get_logger()

## iou计算
def iou_binary_mask(label_pred, label_gt):
  """
  Compute the intersection-over-union score
  :param label_pred:
  :param label_gt:
  :return:
  """
  label = label_gt[:]
  label = label.astype(np.uint8, copy=False)
  label_pred = label_pred.astype(np.uint8, copy=False)

  pixel_union = np.fmax(label, label_pred)#label#这样就只是检测pred中是boundary占gt的boundary中的部分，相当于准确度
  pixel_intersection = np.fmin(label, label_pred)
  pixel_union = pixel_union.ravel()
  pixel_intersection = pixel_intersection.ravel()
  u = np.count_nonzero(pixel_union)
  i = np.count_nonzero(pixel_intersection)
  return float(i + 1E-5) / float(u + 1E-5)


# ## 老代码的生成boundary
# def gen_mask_boundary(mask, dist=10):
#   iterations = 10
#   kernel_ero = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dist, dist))
#   eroded = cv2.erode(mask.astype('uint8'), kernel_ero, iterations)
  
#   boundary = np.zeros(mask.shape)
#   boundary[mask>=1] = 1
#   boundary[eroded>=1] = 0
#   return boundary

#BIOU github代码的生成边界方式 二值化的
def mask_to_boundary(mask, dilation_ratio=0.02):
    """
    Convert binary mask to boundary mask.
    :param mask (numpy array, uint8): binary mask
    :param dilation_ratio (float): ratio to calculate dilation = dilation_ratio * image_diagonal
    :return: boundary mask (numpy array)
    """
    import cv2
    h, w = mask.shape
    img_diag = np.sqrt(h ** 2 + w ** 2)
    dilation = int(round(dilation_ratio * img_diag))
    if dilation < 1:
        dilation = 1
    # Pad image so mask truncated by the image border is also considered as boundary.
    new_mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    kernel = np.ones((3, 3), dtype=np.uint8)
    new_mask_erode = cv2.erode(new_mask, kernel, iterations=dilation)
    mask_erode = new_mask_erode[1 : h + 1, 1 : w + 1]
    # G_d intersects G in the paper.
    return mask - mask_erode

def multi_class_gt_to_boundary(gt, dilation_ratio=0.02):
    gt = torch.tensor(gt).long()
    # gt_one_hot = F.one_hot(gt, 256)[:,:,0:class_num]
    shown_classes = list(np.unique(gt))
    try:
        shown_classes.remove(255)
    except:
        pass
    else:
        pass

    gt_one_hot = F.one_hot(gt, 256)[:,:,shown_classes]
    gt_boundary = torch.ones_like(gt)*255
    if len(shown_classes)<1:
        return np.array(gt_boundary, dtype=np.uint8)
    # class_boundary_list = [mask_to_boundary(np.array(gt_one_hot[:,:,i], dtype=np.uint8)) for i in range(class_num)]
    class_boundary_list = [np.expand_dims(mask_to_boundary(np.array(gt_one_hot[:,:,i], dtype=np.uint8)),axis=0) for i in range(len(shown_classes))]
    class_boundary = np.concatenate(class_boundary_list, axis=0).sum(axis=0)
    # for i in range(class_num):
    #     gt_one_hot[:,:,i]
    class_boundary[class_boundary>1]=1
    return class_boundary

def multi_class_gt_to_multi_boundary(gt, dilation_ratio=0.02, reduce_zero=False):
    gt = torch.tensor(gt).long()
    # gt_one_hot = F.one_hot(gt, 256)[:,:,0:class_num]
    shown_classes = list(np.unique(gt))

    try:
        shown_classes.remove(255)
    except:
        pass
    else:
        pass

    gt_one_hot = F.one_hot(gt, 256)[:,:,shown_classes]
    # gt_boundary = torch.zeros_like(gt)
    # class_boundary_list = [mask_to_boundary(np.array(gt_one_hot[:,:,i], dtype=np.uint8)) for i in range(class_num)]
    # class_boundary_list = [np.expand_dims(mask_to_boundary(np.array(gt_one_hot[:,:,i], dtype=np.uint8)),axis=0) for i in range(len(shown_classes))]
    if reduce_zero:
        class_boundary_list = [np.expand_dims(mask_to_boundary(np.array(gt_one_hot[:,:,i]*shown_classes[i], dtype=np.uint8)),axis=0) for i in range(len(shown_classes))]
    else:
        class_boundary_list = [np.expand_dims(mask_to_boundary(np.array(gt_one_hot[:,:,i]*(shown_classes[i]+1), dtype=np.uint8)),axis=0) for i in range(len(shown_classes))]
    class_boundary = np.concatenate(class_boundary_list, axis=0).sum(axis=0)
    # for i in range(class_num):
    #     gt_one_hot[:,:,i]
    # class_boundary[class_boundary>1]=1
    return class_boundary

def multi_class_gt_to_multi_boundary_pool(args):
    gt, dilation_ratio, reduce_zero=args[0],args[1],args[2]
    gt = torch.tensor(gt).long()
    # gt_one_hot = F.one_hot(gt, 256)[:,:,0:class_num]
    shown_classes = list(np.unique(gt))

    try:
        shown_classes.remove(255)
    except:
        pass
    else:
        pass

    gt_one_hot = F.one_hot(gt, 256)[:,:,shown_classes]
    # gt_boundary = torch.zeros_like(gt)
    # class_boundary_list = [mask_to_boundary(np.array(gt_one_hot[:,:,i], dtype=np.uint8)) for i in range(class_num)]
    # class_boundary_list = [np.expand_dims(mask_to_boundary(np.array(gt_one_hot[:,:,i], dtype=np.uint8)),axis=0) for i in range(len(shown_classes))]
    if reduce_zero:
        class_boundary_list = [np.expand_dims(mask_to_boundary(np.array(gt_one_hot[:,:,i]*shown_classes[i], dtype=np.uint8)),axis=0) for i in range(len(shown_classes))]
    else:
        class_boundary_list = [np.expand_dims(mask_to_boundary(np.array(gt_one_hot[:,:,i]*(shown_classes[i]+1), dtype=np.uint8)),axis=0) for i in range(len(shown_classes))]
    class_boundary = np.concatenate(class_boundary_list, axis=0).sum(axis=0)
    # for i in range(class_num):
    #     gt_one_hot[:,:,i]
    # class_boundary[class_boundary>1]=1
    return class_boundary


## boundary iou
def boundary_iou(pred, gt):
  pd = multi_class_gt_to_boundary(pred)
  gd = multi_class_gt_to_boundary(gt)
  return iou_binary_mask(pd, gd)
