# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/facebookresearch/detr/blob/master/models/detr.py
"""
MaskFormer criterion.
"""
import logging

import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

from detectron2.utils.comm import get_world_size
from detectron2.projects.point_rend.point_features import (
    get_uncertain_point_coords_with_randomness,
    point_sample,
)

from ..utils.misc import is_dist_avail_and_initialized, nested_tensor_from_tensor_list


def dice_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
    ):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks


dice_loss_jit = torch.jit.script(
    dice_loss
)  # type: torch.jit.ScriptModule


def sigmoid_ce_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
    ):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

    return loss.mean(1).sum() / num_masks


sigmoid_ce_loss_jit = torch.jit.script(
    sigmoid_ce_loss
)  # type: torch.jit.ScriptModule


def calculate_uncertainty(logits):
    """
    We estimate uncerainty as L1 distance between 0.0 and the logit prediction in 'logits' for the
        foreground class in `classes`.
    Args:
        logits (Tensor): A tensor of shape (R, 1, ...) for class-specific or
            class-agnostic, where R is the total number of predicted masks in all images and C is
            the number of foreground classes. The values are logits.
    Returns:
        scores (Tensor): A tensor of shape (R, 1, ...) that contains uncertainty scores with
            the most uncertain locations having the highest uncertainty score.
    """
    assert logits.shape[1] == 1
    gt_class_logits = logits.clone()
    return -(torch.abs(gt_class_logits))


class SetCriterion(nn.Module):
    """This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses,
                 num_points, oversample_ratio, importance_sample_ratio):
        """Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)

        # pointwise mask loss parameters
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio

        #########################################################
        # 我加的cbl的参数
        self.weight_dict['loss_context'] = 2.0
        self.weight_dict['loss_NCE'] = 0.4
        self.weight_dict['loss_CN'] = 4
        base_weight = np.array([[1, 1, 1, 1, 1], 
                                [1, 1, 1, 1, 1], 
                                [1, 1, 0, 1, 1], 
                                [1, 1, 1, 1, 1],
                                [1, 1, 1, 1, 1], ])
        base_weight = base_weight.reshape((1, 1, 5, 5))
        self.same_class_extractor_weight = np.repeat(base_weight, 256, axis=0)
        self.same_class_extractor_weight = torch.FloatTensor(self.same_class_extractor_weight)
        # self.same_class_extractor_weight.requires_grad(False)
        self.same_class_number_extractor_weight = base_weight
        self.same_class_number_extractor_weight = torch.FloatTensor(self.same_class_number_extractor_weight)
        #########################################################


    def loss_labels(self, outputs, targets, indices, num_masks):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"].float()

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(
            src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device
        )
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {"loss_ce": loss_ce}
        return losses
    
    def loss_masks(self, outputs, targets, indices, num_masks):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # No need to upsample predictions as we are using normalized coordinates :)
        # N x 1 x H x W
        src_masks = src_masks[:, None]
        target_masks = target_masks[:, None]

        with torch.no_grad():
            # sample point_coords
            point_coords = get_uncertain_point_coords_with_randomness(
                src_masks,
                lambda logits: calculate_uncertainty(logits),
                self.num_points,
                self.oversample_ratio,
                self.importance_sample_ratio,
            )
            # get gt labels
            point_labels = point_sample(
                target_masks,
                point_coords,
                align_corners=False,
            ).squeeze(1)

        point_logits = point_sample(
            src_masks,
            point_coords,
            align_corners=False,
        ).squeeze(1)

        losses = {
            "loss_mask": sigmoid_ce_loss_jit(point_logits, point_labels, num_masks),
            "loss_dice": dice_loss_jit(point_logits, point_labels, num_masks),
        }

        del src_masks
        del target_masks
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_masks):
        loss_map = {
            'labels': self.loss_labels,
            'masks': self.loss_masks,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_masks)

    def forward(self, outputs, targets):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_masks = sum(len(t["labels"]) for t in targets)
        num_masks = torch.as_tensor(
            [num_masks], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_masks)
        num_masks = torch.clamp(num_masks / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_masks))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_masks)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses

    def forward_withcbl(self, outputs, targets, sem_seg_logits, gt_sem, gt_sem_boundary):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_masks = sum(len(t["labels"]) for t in targets)
        num_masks = torch.as_tensor(
            [num_masks], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_masks)
        num_masks = torch.clamp(num_masks / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_masks))
        
        ########################################################################
        # 计算cbl
        losses['loss_context'] = self.context_loss(outputs['mask_features'], seg_label=gt_sem, gt_boundary_seg=gt_sem_boundary)
        losses['loss_NCE'], losses['loss_CN'] = self.er_loss(
                outputs['mask_features'], 
                seg_label=gt_sem, 
                seg_logit=sem_seg_logits, 
                gt_boundary_seg=gt_sem_boundary,
                seg_weight=outputs['mask_embed'],
                cls_logits=outputs['pred_logits'][..., :-1]
            )

        # losses['loss_NCE'] = losses['loss_NCE']*0.2
        # losses['loss_CN'] = losses['loss_CN']*2
        ########################################################################


        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_masks)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses

    def __repr__(self):
        head = "Criterion " + self.__class__.__name__
        body = [
            "matcher: {}".format(self.matcher.__repr__(_repr_indent=8)),
            "losses: {}".format(self.losses),
            "weight_dict: {}".format(self.weight_dict),
            "num_classes: {}".format(self.num_classes),
            "eos_coef: {}".format(self.eos_coef),
            "num_points: {}".format(self.num_points),
            "oversample_ratio: {}".format(self.oversample_ratio),
            "importance_sample_ratio: {}".format(self.importance_sample_ratio),
        ]
        _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)
    
    ################################################################
        # 我加的cbl
    def context_loss(self, er_input, seg_label, gt_boundary_seg, kernel_size=5):
        seg_label = F.interpolate(seg_label.unsqueeze(1).float(), size=er_input.shape[2:], mode='nearest').long()
        gt_boundary_seg = F.interpolate(gt_boundary_seg.unsqueeze(1).float(), size=er_input.shape[2:], mode='nearest').long()
        context_loss_final = torch.tensor(0.0, device=er_input.device)
        context_loss = torch.tensor(0.0, device=er_input.device)
        # 获取参与运算的边缘像素mask gt_b (B,1,H,W)
        gt_b = gt_boundary_seg
        # 将ignore的像素置零，现在gt_b里面的0表示不参与loss计算
        gt_b[gt_b==255]=0
        seg_label_copy = seg_label.clone()
        seg_label_copy[seg_label_copy==255]=0
        gt_b = gt_b*seg_label_copy
        seg_label_one_hot = F.one_hot(seg_label.squeeze(1), num_classes=256)[:,:,:,0:self.num_classes].permute(0,3,1,2)

        b,c,h,w = er_input.shape
        scale_num = b
        for i in range(b):
            cal_mask = (gt_b[i][0]>0).bool()
            if cal_mask.sum()<1:
                scale_num = scale_num-1
                continue

            position = torch.where(gt_b[i][0])
            position_mask = ((kernel_size//2)<=position[0]) * (position[0]<=(er_input.shape[-2]-1-(kernel_size//2))) * ((kernel_size//2)<=position[1]) * (position[1]<=(er_input.shape[-1]-1-(kernel_size//2)))
            position_selected = (position[0][position_mask], position[1][position_mask])
            position_shift_list = []
            for ki in range(kernel_size):
                for kj in range(kernel_size):
                    if ki==kj==(kernel_size//2):
                        continue
                    position_shift_list.append((position_selected[0]+ki-(kernel_size//2),position_selected[1]+kj-(kernel_size//2)))
            # context_loss_batchi = torch.zeros_like(er_input[i].permute(1,2,0)[position_selected][0])
            context_loss_pi = torch.tensor(0.0, device=er_input.device)
            for pi in range(len(position_shift_list)):
                boudary_simi = F.cosine_similarity(er_input[i].permute(1,2,0)[position_selected], er_input[i].permute(1,2,0)[position_shift_list[pi]], dim=1)
                boudary_simi_label = torch.sum(seg_label_one_hot[i].permute(1,2,0)[position_selected] * seg_label_one_hot[i].permute(1,2,0)[position_shift_list[pi]], dim=-1)
                context_loss_pi = context_loss_pi + F.mse_loss(boudary_simi, boudary_simi_label.float())
            context_loss += (context_loss_pi / len(position_shift_list))
        context_loss = context_loss/scale_num
        if torch.isnan(context_loss):
            return context_loss_final
        else:
            return context_loss

    def er_loss(self, er_input, seg_label, seg_logit, gt_boundary_seg, seg_weight, cls_logits):
        shown_class = list(seg_label.unique())
        pred_label = seg_logit.max(dim=1)[1]
        pred_label_one_hot = F.one_hot(pred_label, num_classes=self.num_classes).permute(0,3,1,2)
        seg_label = F.interpolate(seg_label.unsqueeze(1).float(), size=er_input.shape[2:], mode='nearest').long()
        gt_boundary_seg = F.interpolate(gt_boundary_seg.unsqueeze(1).float(), size=er_input.shape[2:], mode='nearest').long()
        # 获取参与运算的边缘像素mask gt_b (B,1,H,W)
        gt_b = gt_boundary_seg
        # 将ignore的像素置零，现在gt_b里面的0表示不参与loss计算
        gt_b[gt_b==255]=0
        edge_mask = gt_b.squeeze(1)
        # 下面按照每个出现的类计算每个类的er loss
        # 首先提取出每个类各自的boundary
        seg_label_one_hot = F.one_hot(seg_label.squeeze(1), num_classes=256)[:,:,:,0:self.num_classes].permute(0,3,1,2)
        if self.same_class_extractor_weight.device!=er_input.device: 
            self.same_class_extractor_weight = self.same_class_extractor_weight.to(er_input.device)
            print("er move:",self.same_class_extractor_weight.device)
        if self.same_class_number_extractor_weight.device!=er_input.device: 
            self.same_class_number_extractor_weight = self.same_class_number_extractor_weight.to(er_input.device)
        # print(self.same_class_number_extractor_weight)
        same_class_extractor = NeighborExtractor5(256)
        same_class_extractor.same_class_extractor.weight.data = self.same_class_extractor_weight
        same_class_number_extractor = NeighborExtractor5(1)
        same_class_number_extractor.same_class_extractor.weight.data = self.same_class_number_extractor_weight
        
        try:
            shown_class.remove(torch.tensor(255))
        except:
            pass
        # er_input = er_input.permute(0,2,3,1)
        neigh_classfication_loss_total = torch.tensor(0.0, device=er_input.device)
        close2neigh_loss_total = torch.tensor(0.0, device=er_input.device)
        cal_class_num = len(shown_class)
        for i in range(len(shown_class)):
            now_class_mask = seg_label_one_hot[:,shown_class[i],:,:]
            now_pred_class_mask = pred_label_one_hot[:,shown_class[i],:,:]
            # er_input 乘当前类的mask，就把所有不是当前类的像素置为0了
            # 得到的now_neighbor_feat是只有当前类的特征
            now_neighbor_feat = same_class_extractor(er_input*now_class_mask.unsqueeze(1))
            now_correct_neighbor_feat = same_class_extractor(er_input*(now_class_mask*now_pred_class_mask).unsqueeze(1))
            # 下面是获得当前类的每个像素邻居中同属当前点的像素个数
            now_class_num_in_neigh = same_class_number_extractor(now_class_mask.unsqueeze(1).float())
            now_correct_class_num_in_neigh = same_class_number_extractor((now_class_mask*now_pred_class_mask).unsqueeze(1).float())
            # 需要得到 可以参与er loss 计算的像素
            # 一个像素若要参与er loss计算，需要具备：
            # 1.邻居中具有同属当前类的像素
            # 2.当前像素要在边界上
            # 如果没有满足这些条件的像素 则直接跳过这个类的计算
            pixel_cal_mask = (now_class_num_in_neigh.squeeze(1)>=1)*(edge_mask.bool()*now_class_mask.bool()).detach()
            pixel_mse_cal_mask = (now_correct_class_num_in_neigh.squeeze(1)>=1)*(edge_mask.bool()*now_class_mask.bool()*now_pred_class_mask.bool()).detach()
            if pixel_cal_mask.sum()<1 or pixel_mse_cal_mask.sum()<1:
                cal_class_num = cal_class_num - 1
                continue            
            # 这里是把邻居特征做平均
            class_forward_feat = now_neighbor_feat/(now_class_num_in_neigh+1e-5)
            class_correct_forward_feat = now_correct_neighbor_feat/(now_correct_class_num_in_neigh+1e-5)

            # 选择出参与loss计算的像素的原始特征
            origin_mse_pixel_feat = er_input.permute(0,2,3,1)[pixel_mse_cal_mask].permute(1,0)
            # 选择出参与loss计算的像素的邻居平均特征
            #############
            # 这个不行了 因为这个每个batch有自己的分类weight
            # neigh_pixel_feat = class_forward_feat.permute(0,2,3,1)[pixel_cal_mask].permute(1,0).unsqueeze(0).unsqueeze(-1)
            # neigh_mse_pixel_feat = class_correct_forward_feat.permute(0,2,3,1)[pixel_mse_cal_mask].permute(1,0).unsqueeze(0).unsqueeze(-1)
            batch_size = er_input.shape[0]
            neigh_pixel_feat = [class_forward_feat.permute(0,2,3,1)[x][pixel_cal_mask[x]].permute(1,0) for x in range(batch_size)]
            neigh_mse_pixel_feat = class_correct_forward_feat.permute(0,2,3,1)[pixel_mse_cal_mask].permute(1,0)
            # neigh_mse_pixel_feat = [class_correct_forward_feat.permute(0,2,3,1)[x][pixel_mse_cal_mask[x]].permute(1,0) for x in range(batch_size)]
            # 邻居平均特征也要能够正确分类，且用同样的分类器才行
            # 这个地方可以试试不detach掉的
            ##############
            # 这个也不行了 因为conv_seg 已经不存在了
            # neigh_pixel_feat_prediction = F.conv2d(neigh_pixel_feat, weight=self.conv_seg.weight.to(neigh_pixel_feat.dtype).detach(), bias=self.conv_seg.bias.to(neigh_pixel_feat.dtype).detach())
            # 这个是第一次改了没改对的 因为b这一维度变成了list 所以这个已经用不了了
            # 要想办法先
            # seg_weight: B Q D    neigh_pixel_feat: B 个 【1 D N 1】
            # seg_weight[i] mul neigh_pixel_feat[i] = 【Q D】*【D N】= [Q N] 
            # 得到下面这个 N个选出来的像素feature关于Q个embed的mask的激活值
            neigh_pixel_feat_mask = [torch.einsum("qd,dn->qn", seg_weight[x], neigh_pixel_feat[x]).sigmoid() for x in range(batch_size)]
            # 然后这个【q n】要跟类别关联起来 就是与q个embed的类别logits做加权
            # 【q,n】*[q,c]=[n,c]
            neigh_pixel_feat_prediction = [torch.einsum("qc,qn->cn",cls_logits[x],neigh_pixel_feat_mask[x]) for x in range(batch_size)]
            neigh_pixel_feat_logits = torch.cat(neigh_pixel_feat_prediction, dim=-1).unsqueeze(0)
            # 为邻居平均特征的分类loss产生GT，即当前类中像素的邻居（因为都是同属当前类的邻居）的标签也是当前类
            # 因此乘shown_class[i]
            # gt_for_neigh_output = shown_class[i]*torch.ones((1, neigh_pixel_feat_logits.shape[2],1)).to(er_input.device).long()
            gt_for_neigh_output = shown_class[i]*torch.ones((1, neigh_pixel_feat_logits.shape[-1])).to(er_input.device).long()

            neigh_classfication_loss = F.cross_entropy(neigh_pixel_feat_logits, gt_for_neigh_output)
            # 当前点的像素 要向周围同类像素的平均特征靠近
            # close2neigh_loss = F.mse_loss(origin_pixel_feat, neigh_pixel_feat.detach())
            close2neigh_loss = F.mse_loss(origin_mse_pixel_feat, neigh_mse_pixel_feat.detach())
            neigh_classfication_loss_total = neigh_classfication_loss_total + neigh_classfication_loss
            close2neigh_loss_total = close2neigh_loss_total + close2neigh_loss
        if cal_class_num==0:
            return neigh_classfication_loss_total, close2neigh_loss_total
        neigh_classfication_loss_total = neigh_classfication_loss_total / cal_class_num
        close2neigh_loss_total = close2neigh_loss_total / cal_class_num
        return neigh_classfication_loss_total, close2neigh_loss_total

    def forward_withcblata(self, outputs, targets, sem_seg_logits, gt_sem, gt_sem_boundary):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_masks = sum(len(t["labels"]) for t in targets)
        num_masks = torch.as_tensor(
            [num_masks], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_masks)
        num_masks = torch.clamp(num_masks / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_masks))
        
        ########################################################################
        # 计算cbl
        losses['loss_context'] = self.context_loss(outputs['mask_features'], seg_label=gt_sem, gt_boundary_seg=gt_sem_boundary)
        losses['loss_NCE'], losses['loss_CN'] = self.er_loss(
                outputs['mask_features'], 
                seg_label=gt_sem, 
                seg_logit=sem_seg_logits, 
                gt_boundary_seg=gt_sem_boundary,
                seg_weight=outputs['mask_embed'],
                cls_logits=outputs['pred_logits'][..., :-1]
            )

        # losses['loss_NCE'] = losses['loss_NCE']*0.2
        # losses['loss_CN'] = losses['loss_CN']*2
        ########################################################################


        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_masks)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses

    def erata_loss(self, er_input, seg_label, seg_logit, gt_boundary_seg, seg_weight, cls_logits):
        b,d,h,w = er_input.shape
        
        shown_class = list(seg_label.unique())
        pred_label = seg_logit.max(dim=1)[1]
        pred_label_one_hot = F.one_hot(pred_label, num_classes=self.num_classes).permute(0,3,1,2)
        seg_label = F.interpolate(seg_label.unsqueeze(1).float(), size=er_input.shape[2:], mode='nearest').long()
        gt_boundary_seg = F.interpolate(gt_boundary_seg.unsqueeze(1).float(), size=er_input.shape[2:], mode='nearest').long()
        # 获取参与运算的边缘像素mask gt_b (B,1,H,W)
        gt_b = gt_boundary_seg
        # 将ignore的像素置零，现在gt_b里面的0表示不参与loss计算
        gt_b[gt_b==255]=0
        edge_mask = gt_b.squeeze(1)
        # 下面按照每个出现的类计算每个类的er loss
        # 首先提取出每个类各自的boundary
        seg_label_one_hot = F.one_hot(seg_label.squeeze(1), num_classes=256)[:,:,:,0:self.num_classes].permute(0,3,1,2)
        if self.same_class_extractor_weight.device!=er_input.device: 
            self.same_class_extractor_weight = self.same_class_extractor_weight.to(er_input.device)
            print("er move:",self.same_class_extractor_weight.device)
        if self.same_class_number_extractor_weight.device!=er_input.device: 
            self.same_class_number_extractor_weight = self.same_class_number_extractor_weight.to(er_input.device)
        # print(self.same_class_number_extractor_weight)
        same_class_extractor = NeighborExtractor5(256)
        same_class_extractor.same_class_extractor.weight.data = self.same_class_extractor_weight
        same_class_number_extractor = NeighborExtractor5(1)
        same_class_number_extractor.same_class_extractor.weight.data = self.same_class_number_extractor_weight
        
        try:
            shown_class.remove(torch.tensor(255))
        except:
            pass
        # er_input = er_input.permute(0,2,3,1)
        neigh_classfication_loss_total = torch.tensor(0.0, device=er_input.device)
        close2neigh_loss_total = torch.tensor(0.0, device=er_input.device)
        cal_class_num = len(shown_class)
        correct_class_center_list = []
        
        for i in range(len(shown_class)):
            now_class_mask = seg_label_one_hot[:,shown_class[i],:,:]
            now_pred_class_mask = pred_label_one_hot[:,shown_class[i],:,:]
            # er_input 乘当前类的mask，就把所有不是当前类的像素置为0了
            # 得到的now_neighbor_feat是只有当前类的特征
            now_neighbor_feat = same_class_extractor(er_input*now_class_mask.unsqueeze(1))
            ####################################################################################
            now_correct_mask = now_class_mask*now_pred_class_mask
            correct_class_center = torch.bmm(er_input.view(b,d,h*w), now_correct_mask.unsqueeze(-1).view(b,h*w,1).to(er_input.dtype)).permute(0,2,1)
            count_weight = now_correct_mask.view(b,-1).sum(-1, keepdim=True).unsqueeze(-1).to(er_input.dtype)
            correct_class_center = correct_class_center/(count_weight+1e-8)
            correct_class_center_list.append(correct_class_center)
            ####################################################################################

            now_correct_neighbor_feat = same_class_extractor(er_input*(now_correct_mask).unsqueeze(1))
            # 下面是获得当前类的每个像素邻居中同属当前点的像素个数
            now_class_num_in_neigh = same_class_number_extractor(now_class_mask.unsqueeze(1).float())
            now_correct_class_num_in_neigh = same_class_number_extractor((now_correct_mask).unsqueeze(1).float())
            # 需要得到 可以参与er loss 计算的像素
            # 一个像素若要参与er loss计算，需要具备：
            # 1.邻居中具有同属当前类的像素
            # 2.当前像素要在边界上
            # 如果没有满足这些条件的像素 则直接跳过这个类的计算
            pixel_cal_mask = (now_class_num_in_neigh.squeeze(1)>=1)*(edge_mask.bool()*now_class_mask.bool()).detach()
            pixel_mse_cal_mask = (now_correct_class_num_in_neigh.squeeze(1)>=1)*(edge_mask.bool()*now_class_mask.bool()*now_pred_class_mask.bool()).detach()
            if pixel_cal_mask.sum()<1 or pixel_mse_cal_mask.sum()<1:
                cal_class_num = cal_class_num - 1
                continue            
            # 这里是把邻居特征做平均
            class_forward_feat = now_neighbor_feat/(now_class_num_in_neigh+1e-5)
            class_correct_forward_feat = now_correct_neighbor_feat/(now_correct_class_num_in_neigh+1e-5)

            # 选择出参与loss计算的像素的原始特征
            origin_mse_pixel_feat = er_input.permute(0,2,3,1)[pixel_mse_cal_mask].permute(1,0)
            # 选择出参与loss计算的像素的邻居平均特征
            #############
            # 这个不行了 因为这个每个batch有自己的分类weight
            # neigh_pixel_feat = class_forward_feat.permute(0,2,3,1)[pixel_cal_mask].permute(1,0).unsqueeze(0).unsqueeze(-1)
            # neigh_mse_pixel_feat = class_correct_forward_feat.permute(0,2,3,1)[pixel_mse_cal_mask].permute(1,0).unsqueeze(0).unsqueeze(-1)
            batch_size = er_input.shape[0]
            neigh_pixel_feat = [class_forward_feat.permute(0,2,3,1)[x][pixel_cal_mask[x]].permute(1,0) for x in range(batch_size)]
            neigh_mse_pixel_feat = class_correct_forward_feat.permute(0,2,3,1)[pixel_mse_cal_mask].permute(1,0)
            # neigh_mse_pixel_feat = [class_correct_forward_feat.permute(0,2,3,1)[x][pixel_mse_cal_mask[x]].permute(1,0) for x in range(batch_size)]
            # 邻居平均特征也要能够正确分类，且用同样的分类器才行
            # 这个地方可以试试不detach掉的
            ##############
            # 这个也不行了 因为conv_seg 已经不存在了
            # neigh_pixel_feat_prediction = F.conv2d(neigh_pixel_feat, weight=self.conv_seg.weight.to(neigh_pixel_feat.dtype).detach(), bias=self.conv_seg.bias.to(neigh_pixel_feat.dtype).detach())
            # 这个是第一次改了没改对的 因为b这一维度变成了list 所以这个已经用不了了
            # 要想办法先
            # seg_weight: B Q D    neigh_pixel_feat: B 个 【1 D N 1】
            # seg_weight[i] mul neigh_pixel_feat[i] = 【Q D】*【D N】= [Q N] 
            # 得到下面这个 N个选出来的像素feature关于Q个embed的mask的激活值
            neigh_pixel_feat_mask = [torch.einsum("qd,dn->qn", seg_weight[x], neigh_pixel_feat[x]).sigmoid() for x in range(batch_size)]
            # 然后这个【q n】要跟类别关联起来 就是与q个embed的类别logits做加权
            # 【q,n】*[q,c]=[n,c]
            neigh_pixel_feat_prediction = [torch.einsum("qc,qn->cn",cls_logits[x],neigh_pixel_feat_mask[x]) for x in range(batch_size)]
            neigh_pixel_feat_logits = torch.cat(neigh_pixel_feat_prediction, dim=-1).unsqueeze(0)
            # 为邻居平均特征的分类loss产生GT，即当前类中像素的邻居（因为都是同属当前类的邻居）的标签也是当前类
            # 因此乘shown_class[i]
            # gt_for_neigh_output = shown_class[i]*torch.ones((1, neigh_pixel_feat_logits.shape[2],1)).to(er_input.device).long()
            gt_for_neigh_output = shown_class[i]*torch.ones((1, neigh_pixel_feat_logits.shape[-1])).to(er_input.device).long()

            neigh_classfication_loss = F.cross_entropy(neigh_pixel_feat_logits, gt_for_neigh_output)
            # 当前点的像素 要向周围同类像素的平均特征靠近
            # close2neigh_loss = F.mse_loss(origin_pixel_feat, neigh_pixel_feat.detach())
            close2neigh_loss = F.mse_loss(origin_mse_pixel_feat, neigh_mse_pixel_feat.detach())

            ########################################################################################################
            correct_global_class_centers = torch.cat(correct_class_center_list, dim=1).contiguous()# correct_global_class_centers = torch.bmm(viewed_now_correct_mask, viewed_er_input)/(viewed_now_correct_mask.sum(-1, keepdim=True)+1e-8)
            ata_mask = 1. - torch.eye(len(shown_class), device=er_input.device).unsqueeze(0).repeat(er_input.shape[0],1,1)# ata_mask = 1. - torch.eye(len(shown_class), device=er_input.device)
            # dist_matrix = [torch.mm(correct_global_class_centers.view(er_input.shape[0], len(shown_class), er_input.shape[1])[bi], 
            #                         correct_global_class_centers.view(er_input.shape[0], len(shown_class), er_input.shape[1])[bi].T).mul(ata_mask).abs() 
            #                                                                         for bi in range(0, er_input.shape[0])]
            dist_matrix = torch.bmm(correct_global_class_centers.view(er_input.shape[0], len(shown_class), er_input.shape[1]), 
                        correct_global_class_centers.view(er_input.shape[0], len(shown_class), er_input.shape[1]).permute(0,2,1)).mul(ata_mask)
            cosdist = torch.cat([dist_matrix[bi].mean().unsqueeze(0) for bi in range(0, er_input.shape[0])], dim=0)
            ata_loss = cosdist[~torch.isnan(cosdist)].mean()
            ########################################################################################################

            neigh_classfication_loss_total = neigh_classfication_loss_total + neigh_classfication_loss
            close2neigh_loss_total = close2neigh_loss_total + close2neigh_loss
        if cal_class_num==0:
            return neigh_classfication_loss_total, close2neigh_loss_total
        neigh_classfication_loss_total = neigh_classfication_loss_total / cal_class_num
        close2neigh_loss_total = close2neigh_loss_total / cal_class_num
        return neigh_classfication_loss_total, close2neigh_loss_total


class NeighborExtractor5(nn.Module):
    def __init__(self, input_channel):
        super(NeighborExtractor5, self).__init__()
        same_class_neighbor = np.array([[1., 1., 1., 1., 1.], 
                                        [1., 1., 1., 1., 1.], 
                                        [1., 1., 0., 1., 1.], 
                                        [1., 1., 1., 1., 1.],
                                        [1., 1., 1., 1., 1.], ])
        same_class_neighbor = same_class_neighbor.reshape((1, 1, 5, 5))
        same_class_neighbor = np.repeat(same_class_neighbor, input_channel, axis=0)
        self.same_class_extractor = nn.Conv2d(input_channel, input_channel, kernel_size=5, padding=2, bias=False, groups=input_channel)
        self.same_class_extractor.weight.data = torch.from_numpy(same_class_neighbor)
    
    def forward(self, feat):
        output = self.same_class_extractor(feat)
        return output
