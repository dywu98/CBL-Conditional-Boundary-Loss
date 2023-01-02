import os.path as osp

import numpy as np
from mmseg.core.evaluation import multi_class_gt_to_boundary

from ..builder import PIPELINES

@PIPELINES.register_module()
class GenerateBoundary(object):
    """Load annotations for semantic segmentation.

    Args:
        reduce_zero_label (bool): Whether reduce all label value by 1.
            Usually used for datasets where 0 is background label.
            Default: False.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
        imdecode_backend (str): Backend for :func:`mmcv.imdecode`. Default:
            'pillow'
    """

    def __init__(self,
                 dilation=0.02):
        self.dilation = dilation

    def __call__(self, results):
        """Call function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`mmseg.CustomDataset`.

        Returns:
            dict: The dict contains loaded semantic segmentation annotations.
        """
        results['gt_boundary_seg'] = multi_class_gt_to_boundary(results['gt_semantic_seg'], self.dilation)
        results['gt_boundary_seg'][results['gt_semantic_seg']==255]=255
        results['seg_fields'].append('gt_boundary_seg')
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(dilation={self.dilation},'
        return repr_str
