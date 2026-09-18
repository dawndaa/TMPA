"""Remote-sensing datasets with DAF corruption injected before preprocessing."""

from __future__ import annotations

import os

import numpy as np
import torch
from PIL import Image
from mmcv.transforms import Compose

from data.datautils import build_dataset_remote, load_seg_map
from .corruptions import apply_corruption


class CorruptedRemoteDataset(build_dataset_remote):
    """TMPA remote dataset with an on-the-fly DAF corruption domain."""

    def __init__(
        self,
        *args,
        corruption='original',
        corruption_severity=5,
        daf_root=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.corruption = corruption
        self.corruption_severity = int(corruption_severity)
        self.daf_root = daf_root

    def __getitem__(self, idx):
        image_path = self.image_list[idx]
        image_name = os.path.basename(image_path).split('.')[0]
        image = Image.open(image_path).convert('RGB')
        ori_shape = image.size

        image_np = np.array(image, dtype=np.uint8)
        image_np = apply_corruption(
            image_np,
            corruption_name=self.corruption,
            severity=self.corruption_severity,
            sample_idx=idx,
            daf_root=self.daf_root,
        )

        results = dict(img=image_np)
        resize_transform = Compose([self.resize])
        results = resize_transform(results)
        image = results['img']

        if self.transform:
            image = self.transform(image)

        if self.set_id == 'iSAID':
            mask = torch.tensor(
                load_seg_map(
                    self.label_list[idx].split('.')[0] + '_instance_color_RGB.png',
                    reduce_zero_label=False,
                )
            ).long()
        elif self.set_id in ['UDD5', 'UAVid', 'VDD']:
            mask = torch.tensor(
                load_seg_map(
                    self.label_list[idx].split('.')[0] + '.png',
                    reduce_zero_label=False,
                )
            ).long()
        elif self.set_id in ['CHN6-CUG', 'DeepGlobe', 'GlobalRoadSet_Val']:
            mask = torch.tensor(
                load_seg_map(
                    self.label_list[idx].split('.')[0] + '.png',
                    reduce_zero_label=False,
                )
            ).long()
        elif self.set_id in ['LoveDA', 'potsdam', 'vaihingen']:
            mask = torch.tensor(load_seg_map(self.label_list[idx])).long()
        else:
            mask = torch.tensor(
                load_seg_map(self.label_list[idx], reduce_zero_label=False)
            ).long()

        return image_name, image, ori_shape, mask
