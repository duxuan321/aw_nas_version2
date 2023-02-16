import copy
import pickle

import numpy as np
from skimage import io

from .pcdet.datasets.kitti import kitti_utils
from .pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils
from .pcdet.utils import box_utils, calibration_kitti, common_utils, object3d_kitti
from .pcdet.datasets.dataset import DatasetTemplate
from .pcdet.datasets.kitti.kitti_dataset import KittiDataset
from .pcdet.datasets.common_3d_ap import eval_map as eval_map_simple_3d
from .pcdet.datasets.common_3d_ap import eval_bev_map as eval_map_simple_bev
from .pcdet.datasets import DistributedSampler as MyDistributedSampler
from aw_nas.dataset.base import BaseDataset

class Kitti(BaseDataset):
    NAME = "kitti"

    def __init__(self, cfg, class_names, root_path=None, logger=None):
        super(Kitti, self).__init__()

        self.datasets = {}
        self.datasets["train"] = KittiDataset(
            dataset_cfg=cfg,
            class_names=class_names,
            root_path=root_path,
            training=True,
            logger=logger,
        )

        self.datasets["test"] = KittiDataset(
            dataset_cfg=cfg,
            class_names=class_names,
            root_path=root_path,
            training=False,
            logger=logger,
        )

    def splits(self):
        return self.datasets

    @classmethod
    def data_type(cls):
        return "image"
    