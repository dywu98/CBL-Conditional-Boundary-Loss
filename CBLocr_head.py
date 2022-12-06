import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule

from mmseg.ops import resize
from ..builder import HEADS
from ..utils import SelfAttentionBlock as _SelfAttentionBlock
from .cascade_decode_head import BaseCascadeDecodeHead

from .condconv import ConditionalFilterLayer, ATAConditionalFilterLayer
from mmcv.runner import auto_fp16, force_fp32
from ..losses import accuracy
from ..builder import build_loss

import numpy as np

class SpatialGatherModule(nn.Module):
    """Aggregate the context features according to the initial predicted
    probability distribution.

    Employ the soft-weighted method to aggregate the context.
    """

    def __init__(self, scale):
        super(SpatialGatherModule, self).__init__()
        self.scale = scale

    def forward(self, feats, probs):
        """Forward function."""
        batch_size, num_classes, height, width = probs.size()
        channels = feats.size(1)
        probs = probs.view(batch_size, num_classes, -1)
        feats = feats.view(batch_size, channels, -1)
        # [batch_size, height*width, num_classes]
        feats = feats.permute(0, 2, 1)
        # [batch_size, channels, height*width]
        probs = F.softmax(self.scale * probs, dim=2)
        # [batch_size, channels, num_classes]
        ocr_context = torch.matmul(probs, feats)
        ocr_context = ocr_context.permute(0, 2, 1).contiguous().unsqueeze(3)
        return ocr_context


class ObjectAttentionBlock(_SelfAttentionBlock):
    """Make a OCR used SelfAttentionBlock."""

    def __init__(self, in_channels, channels, scale, conv_cfg, norm_cfg,
                 act_cfg):
        if scale > 1:
            query_downsample = nn.MaxPool2d(kernel_size=scale)
        else:
            query_downsample = None
        super(ObjectAttentionBlock, self).__init__(
            key_in_channels=in_channels,
            query_in_channels=in_channels,
            channels=channels,
            out_channels=in_channels,
            share_key_query=False,
            query_downsample=query_downsample,
            key_downsample=None,
            key_query_num_convs=2,
            key_query_norm=True,
            value_out_num_convs=1,
            value_out_norm=True,
            matmul_norm=True,
            with_out=True,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.bottleneck = ConvModule(
            in_channels * 2,
            in_channels,
            1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def forward(self, query_feats, key_feats):
        """Forward function."""
        context = super(ObjectAttentionBlock,
                        self).forward(query_feats, key_feats)
        output = self.bottleneck(torch.cat([context, query_feats], dim=1))
        if self.query_downsample is not None:
            output = resize(query_feats)

        return output


@HEADS.register_module()
class OCRHead(BaseCascadeDecodeHead):
    """Object-Contextual Representations for Semantic Segmentation.

    This head is the implementation of `OCRNet
    <https://arxiv.org/abs/1909.11065>`_.

    Args:
        ocr_channels (int): The intermediate channels of OCR block.
        scale (int): The scale of probability map in SpatialGatherModule in
            Default: 1.
    """

    def __init__(self, ocr_channels, scale=1, **kwargs):
        super(OCRHead, self).__init__(**kwargs)
        self.ocr_channels = ocr_channels
        self.scale = scale
        self.object_context_block = ObjectAttentionBlock(
            self.channels,
            self.ocr_channels,
            self.scale,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        self.spatial_gather_module = SpatialGatherModule(self.scale)

        self.bottleneck = ConvModule(
            self.in_channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def forward(self, inputs, prev_output):
        """Forward function."""
        x = self._transform_inputs(inputs)
        feats = self.bottleneck(x)
        context = self.spatial_gather_module(feats, prev_output)
        object_context = self.object_context_block(feats, context)
        output = self.cls_seg(object_context)

        return output

@HEADS.register_module()
class OCRHead_ATA(BaseCascadeDecodeHead):
    """Object-Contextual Representations for Semantic Segmentation.

    This head is the implementation of `OCRNet
    <https://arxiv.org/abs/1909.11065>`_.

    Args:
        ocr_channels (int): The intermediate channels of OCR block.
        scale (int): The scale of probability map in SpatialGatherModule in
            Default: 1.
    """

    def __init__(self, ocr_channels, scale=1, ata_weight=1.0, dice_config=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4), **kwargs):
        super(OCRHead_ATA, self).__init__(**kwargs)
        self.ocr_channels = ocr_channels
        self.scale = scale
        self.ata_weight = ata_weight
        self.object_context_block = ObjectAttentionBlock(
            self.channels,
            self.ocr_channels,
            self.scale,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        self.spatial_gather_module = SpatialGatherModule(self.scale)

        self.bottleneck = ConvModule(
            self.in_channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        self.cf_layer = ATAConditionalFilterLayer(
            self.channels,
            self.num_classes,
            dice_config,
            self.conv_cfg,
            self.norm_cfg,
            self.act_cfg)

    def forward(self, inputs, prev_output, train=False):
        """Forward function."""
        x = self._transform_inputs(inputs)
        feats = self.bottleneck(x)
        context = self.spatial_gather_module(feats, prev_output)
        object_context = self.object_context_block(feats, context)
        # output = self.cls_seg(object_context)
        output, mask, filters, b = self.cf_layer(object_context, back_mask=prev_output)
        # output = self.cls_seg(output)
        if train:
            return output, mask, filters, b
        else:
            return output

    def losses(self, seg_logit, seg_mask, seg_label, filters, b):
        """Compute segmentation loss."""
        loss = dict()
        seg_logit = resize(
            input=seg_logit,
            size=seg_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        seg_mask = resize(
            input=seg_mask,
            size=seg_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        if self.sampler is not None:
            seg_weight = self.sampler.sample(seg_logit, seg_label)
        else:
            seg_weight = None
        seg_label = seg_label.squeeze(1)
        loss['loss_seg'] = self.loss_decode(
            seg_logit,
            seg_label,
            weight=seg_weight,
            ignore_index=self.ignore_index)
            # ignore_index=self.ignore_index)
        # loss['loss_edge'] = self.edge_loss(
        #     seg_logit,
        #     seg_label,
        #     weight=seg_weight
        # )
        loss['loss_ata'] = self.ata_weight * self.cf_layer.ata_loss(filters, seg_label, self.num_classes, b)
        # loss['loss_dice_mask'] = self.cf_layer.dice_loss(
        #     seg_mask,
        #     seg_label)
        loss['acc_seg'] = accuracy(seg_logit, seg_label)
        # loss['acc_seg'] = accuracy(seg_logit, seg_label)
        return loss

    def forward_train(self, inputs, prev_output, img_metas, gt_semantic_seg,
                      train_cfg):
        """Forward function for training.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
            prev_output (Tensor): The output of previous decode head.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.
            train_cfg (dict): The training config.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        seg_logits, mask, filters, b = self.forward(inputs, prev_output, train=True)
        losses = self.losses(seg_logits, mask, gt_semantic_seg, filters, b)

        return losses
    def forward_test(self, inputs, prev_output, img_metas, test_cfg):
        """Forward function for testing.

        Args:
            inputs (list[Tensor]): List of multi-level img features.
            prev_output (Tensor): The output of previous decode head.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            test_cfg (dict): The testing config.

        Returns:
            Tensor: Output segmentation map.
        """
        return self.forward(inputs, prev_output)

class NeighborExtractor5(nn.Module):
    def __init__(self, input_channel):
        super(NeighborExtractor5, self).__init__()
        same_class_neighbor = np.array([[1, 1, 1, 1, 1], 
                                        [1, 1, 1, 1, 1], 
                                        [1, 1, 0, 1, 1], 
                                        [1, 1, 1, 1, 1],
                                        [1, 1, 1, 1, 1], ], dtype='float32')
        same_class_neighbor = same_class_neighbor.reshape((1, 1, 5, 5))
        same_class_neighbor = np.repeat(same_class_neighbor, input_channel, axis=0)
        self.same_class_extractor = nn.Conv2d(input_channel, input_channel, kernel_size=5, padding=2, bias=False, groups=input_channel)
        self.same_class_extractor.weight.data = torch.from_numpy(same_class_neighbor)
    
    def forward(self, feat):
        output = self.same_class_extractor(feat)
        return output

@HEADS.register_module()
class ER5OCRHead(BaseCascadeDecodeHead):
    """Object-Contextual Representations for Semantic Segmentation.

    This head is the implementation of `OCRNet
    <https://arxiv.org/abs/1909.11065>`_.

    Args:
        ocr_channels (int): The intermediate channels of OCR block.
        scale (int): The scale of probability map in SpatialGatherModule in
            Default: 1.
    """

    def __init__(self, ocr_channels, scale=1, **kwargs):
        super(ER5OCRHead, self).__init__(**kwargs)
        self.ocr_channels = ocr_channels
        self.scale = scale
        self.object_context_block = ObjectAttentionBlock(
            self.channels,
            self.ocr_channels,
            self.scale,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        self.spatial_gather_module = SpatialGatherModule(self.scale)

        self.bottleneck = ConvModule(
            self.in_channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        base_weight = np.array([[1, 1, 1, 1, 1], 
                                [1, 1, 1, 1, 1], 
                                [1, 1, 0, 1, 1], 
                                [1, 1, 1, 1, 1],
                                [1, 1, 1, 1, 1], ], dtype='float32')
        base_weight = base_weight.reshape((1, 1, 5, 5))
        self.same_class_extractor_weight = np.repeat(base_weight, 512, axis=0)
        self.same_class_extractor_weight = torch.FloatTensor(self.same_class_extractor_weight)
        # self.same_class_extractor_weight.requires_grad(False)
        self.same_class_number_extractor_weight = base_weight
        self.same_class_number_extractor_weight = torch.FloatTensor(self.same_class_number_extractor_weight)
        # self.same_class_number_extractor_weight.requires_grad(False)

    def forward(self, inputs, prev_output):
        """Forward function."""
        x = self._transform_inputs(inputs)
        feats = self.bottleneck(x)
        context = self.spatial_gather_module(feats, prev_output)
        object_context = self.object_context_block(feats, context)
        output = self.cls_seg(object_context)

        # return output, object_context
        return output, feats
    
    # def forward_train(self, inputs, img_metas, gt_semantic_seg, train_cfg, **kwargs):
    #     seg_logits, output_er = self.forward(inputs)
    #     losses = self.losses(seg_logits, output_er, gt_semantic_seg, kwargs['gt_boundary_seg'])
    #     return losses
    
    # def forward_test(self, inputs, img_metas, test_cfg):
    #     seg_logits, edge_logits = self.forward(inputs)
    #     return seg_logits

    def forward_train(self, inputs, prev_output, img_metas, gt_semantic_seg,
                      train_cfg, **kwargs):
        seg_logits, output_er = self.forward(inputs, prev_output)
        # losses = self.losses(seg_logits, gt_semantic_seg)
        losses = self.losses(seg_logits, output_er, gt_semantic_seg, kwargs['gt_boundary_seg'])
        return losses

    def forward_test(self, inputs, prev_output, img_metas, test_cfg):
        seg_logits, output_er = self.forward(inputs, prev_output)
        return seg_logits

    @force_fp32(apply_to=('seg_logit', 'output_er'))
    def losses(self, seg_logit, output_er, seg_label, gt_boundary_seg):
        """Compute segmentation loss."""
        loss = dict()
        loss['loss_context'] = self.context_loss(output_er, seg_label, gt_boundary_seg)
        loss['loss_NCE'], loss['loss_CN'] = self.er_loss(output_er, seg_label, seg_logit, gt_boundary_seg)
        loss['loss_NCE'] = loss['loss_NCE']*0.2
        loss['loss_CN'] = loss['loss_CN']*2
        # loss['loss_context'] = self.context_loss(output_er, seg_label, gt_boundary_seg)
        seg_logit = resize(
            input=seg_logit,
            size=seg_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        if self.sampler is not None:
            seg_weight = self.sampler.sample(seg_logit, seg_label)
        else:
            seg_weight = None
        seg_label = seg_label.squeeze(1)
        loss['loss_seg'] = self.loss_decode(
            seg_logit,
            seg_label,
            weight=seg_weight,
            ignore_index=self.ignore_index)
        loss['acc_seg'] = accuracy(seg_logit, seg_label)

        return loss

    def context_loss(self, er_input, seg_label, gt_boundary_seg, kernel_size=5):
        seg_label = F.interpolate(seg_label.float(), size=er_input.shape[2:], mode='nearest').long()
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

    def er_loss(self, er_input, seg_label, seg_logit, gt_boundary_seg):
        shown_class = list(seg_label.unique())
        pred_label = seg_logit.max(dim=1)[1]
        pred_label_one_hot = F.one_hot(pred_label, num_classes=self.num_classes).permute(0,3,1,2)
        seg_label = F.interpolate(seg_label.float(), size=er_input.shape[2:], mode='nearest').long()
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
        same_class_extractor = NeighborExtractor5(512)
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
            # origin_pixel_feat = er_input.permute(0,2,3,1)[pixel_cal_mask].permute(1,0).unsqueeze(0).unsqueeze(-1)
            origin_mse_pixel_feat = er_input.permute(0,2,3,1)[pixel_mse_cal_mask].permute(1,0).unsqueeze(0).unsqueeze(-1)
            # 选择出参与loss计算的像素的邻居平均特征
            neigh_pixel_feat = class_forward_feat.permute(0,2,3,1)[pixel_cal_mask].permute(1,0).unsqueeze(0).unsqueeze(-1)
            neigh_mse_pixel_feat = class_correct_forward_feat.permute(0,2,3,1)[pixel_mse_cal_mask].permute(1,0).unsqueeze(0).unsqueeze(-1)
            # 邻居平均特征也要能够正确分类，且用同样的分类器才行
            # 这个地方可以试试不detach掉的
            # neigh_pixel_feat_prediction = F.conv2d(neigh_pixel_feat, weight=self.conv_seg.weight.to(neigh_pixel_feat.dtype).detach(), bias=self.conv_seg.bias.to(neigh_pixel_feat.dtype).detach())
            # neigh_pixel_feat_prediction = F.conv2d(neigh_pixel_feat, weight=self.conv_seg.weight.clone().to(neigh_pixel_feat.dtype), bias=self.conv_seg.bias.clone().to(neigh_pixel_feat.dtype))
            neigh_pixel_feat_prediction = self.cls_seg(neigh_pixel_feat)
            # 为邻居平均特征的分类loss产生GT，即当前类中像素的邻居（因为都是同属当前类的邻居）的标签也是当前类
            # 因此乘shown_class[i]
            gt_for_neigh_output = shown_class[i]*torch.ones((1,neigh_pixel_feat_prediction.shape[2],1)).to(er_input.device).long()
            neigh_classfication_loss = F.cross_entropy(neigh_pixel_feat_prediction, gt_for_neigh_output)
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

@HEADS.register_module()
class New_ER5OCRHead(BaseCascadeDecodeHead):
    """Object-Contextual Representations for Semantic Segmentation.

    This head is the implementation of `OCRNet
    <https://arxiv.org/abs/1909.11065>`_.

    Args:
        ocr_channels (int): The intermediate channels of OCR block.
        scale (int): The scale of probability map in SpatialGatherModule in
            Default: 1.
    """

    def __init__(self, ocr_channels, scale=1, **kwargs):
        super(New_ER5OCRHead, self).__init__(**kwargs)
        self.ocr_channels = ocr_channels
        self.scale = scale
        self.object_context_block = ObjectAttentionBlock(
            self.channels,
            self.ocr_channels,
            self.scale,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        self.spatial_gather_module = SpatialGatherModule(self.scale)

        self.bottleneck = ConvModule(
            self.in_channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        base_weight = np.array([[1, 1, 1, 1, 1], 
                                [1, 1, 1, 1, 1], 
                                [1, 1, 0, 1, 1], 
                                [1, 1, 1, 1, 1],
                                [1, 1, 1, 1, 1], ], dtype='float32')
        base_weight = base_weight.reshape((1, 1, 5, 5))
        self.same_class_extractor_weight = np.repeat(base_weight, 512, axis=0)
        self.same_class_extractor_weight = torch.FloatTensor(self.same_class_extractor_weight)
        # self.same_class_extractor_weight.requires_grad(False)
        self.same_class_number_extractor_weight = base_weight
        self.same_class_number_extractor_weight = torch.FloatTensor(self.same_class_number_extractor_weight)
        # self.same_class_number_extractor_weight.requires_grad(False)

    def forward(self, inputs, prev_output):
        """Forward function."""
        x = self._transform_inputs(inputs)
        feats = self.bottleneck(x)
        context = self.spatial_gather_module(feats, prev_output)
        object_context = self.object_context_block(feats, context)
        output = self.cls_seg(object_context)

        # return output, object_context
        return output, feats
    
    # def forward_train(self, inputs, img_metas, gt_semantic_seg, train_cfg, **kwargs):
    #     seg_logits, output_er = self.forward(inputs)
    #     losses = self.losses(seg_logits, output_er, gt_semantic_seg, kwargs['gt_boundary_seg'])
    #     return losses
    
    # def forward_test(self, inputs, img_metas, test_cfg):
    #     seg_logits, edge_logits = self.forward(inputs)
    #     return seg_logits

    def forward_train(self, inputs, prev_output, img_metas, gt_semantic_seg,
                      train_cfg, **kwargs):
        seg_logits, output_er = self.forward(inputs, prev_output)
        # losses = self.losses(seg_logits, gt_semantic_seg)
        losses = self.losses(seg_logits, output_er, gt_semantic_seg, kwargs['gt_boundary_seg'])
        return losses

    def forward_test(self, inputs, prev_output, img_metas, test_cfg):
        seg_logits, output_er = self.forward(inputs, prev_output)
        return seg_logits

    @force_fp32(apply_to=('seg_logit', 'output_er'))
    def losses(self, seg_logit, output_er, seg_label, gt_boundary_seg):
        """Compute segmentation loss."""
        loss = dict()
        loss['loss_context'] = self.context_loss(output_er, seg_label, gt_boundary_seg)
        loss['loss_NCE'], loss['loss_CN'] = self.er_loss(output_er, seg_label, seg_logit, gt_boundary_seg)
        loss['loss_NCE'] = loss['loss_NCE']*0.2
        loss['loss_CN'] = loss['loss_CN']*2
        # loss['loss_context'] = self.context_loss(output_er, seg_label, gt_boundary_seg)
        seg_logit = resize(
            input=seg_logit,
            size=seg_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        if self.sampler is not None:
            seg_weight = self.sampler.sample(seg_logit, seg_label)
        else:
            seg_weight = None
        seg_label = seg_label.squeeze(1)
        loss['loss_seg'] = self.loss_decode(
            seg_logit,
            seg_label,
            weight=seg_weight,
            ignore_index=self.ignore_index)
        loss['acc_seg'] = accuracy(seg_logit, seg_label)

        return loss

    def context_loss(self, er_input, seg_label, gt_boundary_seg, kernel_size=5):
        seg_label = F.interpolate(seg_label.float(), size=er_input.shape[2:], mode='nearest').long()
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

    def er_loss(self, er_input, seg_label, seg_logit, gt_boundary_seg):
        shown_class = list(seg_label.unique())
        pred_label = seg_logit.max(dim=1)[1]
        pred_label_one_hot = F.one_hot(pred_label, num_classes=self.num_classes).permute(0,3,1,2)
        seg_label = F.interpolate(seg_label.float(), size=er_input.shape[2:], mode='nearest').long()
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
        same_class_extractor = NeighborExtractor5(512)
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
            # origin_pixel_feat = er_input.permute(0,2,3,1)[pixel_cal_mask].permute(1,0).unsqueeze(0).unsqueeze(-1)
            origin_mse_pixel_feat = er_input.permute(0,2,3,1)[pixel_mse_cal_mask].permute(1,0).unsqueeze(0).unsqueeze(-1)
            # 选择出参与loss计算的像素的邻居平均特征
            neigh_pixel_feat = class_forward_feat.permute(0,2,3,1)[pixel_cal_mask].permute(1,0).unsqueeze(0).unsqueeze(-1)
            neigh_mse_pixel_feat = class_correct_forward_feat.permute(0,2,3,1)[pixel_mse_cal_mask].permute(1,0).unsqueeze(0).unsqueeze(-1)
            # 邻居平均特征也要能够正确分类，且用同样的分类器才行
            # 这个地方可以试试不detach掉的
            neigh_pixel_feat_prediction = F.conv2d(neigh_pixel_feat, weight=self.conv_seg.weight.to(neigh_pixel_feat.dtype).detach(), bias=self.conv_seg.bias.to(neigh_pixel_feat.dtype).detach())
            # 为邻居平均特征的分类loss产生GT，即当前类中像素的邻居（因为都是同属当前类的邻居）的标签也是当前类
            # 因此乘shown_class[i]
            gt_for_neigh_output = shown_class[i]*torch.ones((1,neigh_pixel_feat_prediction.shape[2],1)).to(er_input.device).long()
            neigh_classfication_loss = F.cross_entropy(neigh_pixel_feat_prediction, gt_for_neigh_output)
            # 当前点的像素 要向周围同类像素的平均特征靠近
            # close2neigh_loss = F.mse_loss(origin_pixel_feat, neigh_pixel_feat.detach())
            neigh_mse_pixel_feat_prediction = F.conv2d(neigh_mse_pixel_feat, weight=self.conv_seg.weight.to(neigh_pixel_feat.dtype).detach(), bias=self.conv_seg.bias.to(neigh_pixel_feat.dtype).detach())
            gt_for_neigh_mse_output = shown_class[i]*torch.ones((1,neigh_mse_pixel_feat_prediction.shape[2],1)).to(er_input.device).long()
            neigh_classfication_loss = neigh_classfication_loss + F.cross_entropy(neigh_mse_pixel_feat_prediction, gt_for_neigh_mse_output)
            

            close2neigh_loss = F.mse_loss(origin_mse_pixel_feat, neigh_mse_pixel_feat.detach())
            neigh_classfication_loss_total = neigh_classfication_loss_total + neigh_classfication_loss
            close2neigh_loss_total = close2neigh_loss_total + close2neigh_loss
        if cal_class_num==0:
            return neigh_classfication_loss_total, close2neigh_loss_total
        neigh_classfication_loss_total = neigh_classfication_loss_total / cal_class_num
        close2neigh_loss_total = close2neigh_loss_total / cal_class_num
        return neigh_classfication_loss_total, close2neigh_loss_total