import os
from typing import Tuple
from PIL import Image
import numpy as np

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

from data.fewshot_datasets import *
import data.augmix_ops as augmentations


import warnings
from typing import Dict, Optional, Union

import mmcv   
import mmengine.fileio as fileio   
import torchvision.transforms.functional as TF  
import torch.nn.functional as F  
import math  
import shutil
from torchvision.utils import save_image   

from mmcv.transforms import Compose, Resize


ID_to_DIRNAME={
    'I': 'ImageNet/imagenet',
    'A': 'imagenet-a',
    'K': 'sketch',
    'R': 'imagenet-r',
    'V': 'imagenetv2-matched-frequency-format-val',
    'flower102': 'Flower102',
    'dtd': 'DTD',
    'pets': 'OxfordPets',
    'cars': 'StanfordCars',
    'ucf101': 'UCF101',
    'caltech101': 'Caltech101',
    'food101': 'Food101',
    'sun397': 'SUN397',
    'aircraft': 'fgvc_aircraft',
    'eurosat': 'eurosat',
}

Remote_ID_to_DIRNAME={
    "openearthmap": "OpenEarthMap",
    'loveda': 'LoveDA',
    'isaid': 'iSAID',
    'potsdam': 'potsdam',
    "uavid":"UAVid",
    "udd5":"UDD5",
    "vaihingen": "vaihingen",
    "vdd":"VDD",
    "whu_aerial":"WHU-BD",
    "whu_sat":"WHU_Sat",
    "inria":"Inria",
    "xbd":"xBD",
    "chn6-cug":"CHN6-CUG",
    "deepglobe":"DeepGlobe",
    "massachusetts":"GlobalRoadSet_Val",
    "spacenet":"GlobalRoadSet_Val",
    "wbs_si":"WBS-SI",
}  


def load_seg_map(ann_path, reduce_zero_label=True, imdecode_backend = 'pillow', backend_args=None):
    """Private function to load semantic segmentation annotations.

    Args:
        results (dict): Result dict from :obj:``mmcv.BaseDataset``.

    Returns:
        dict: The dict contains loaded semantic segmentation annotations.
    """
    img_bytes = fileio.get(
        ann_path, backend_args)
    gt_semantic_seg = mmcv.imfrombytes(
        img_bytes, flag='unchanged',
        backend=imdecode_backend).squeeze().astype(np.uint8)

    # # reduce zero_label
    # if reduce_zero_label is None:
    #     reduce_zero_label = results['reduce_zero_label']
    # assert reduce_zero_label == results['reduce_zero_label'], \
    #     'Initialize dataset with `reduce_zero_label` as ' \
    #     f'{results["reduce_zero_label"]} but when load annotation ' \
    #     f'the `reduce_zero_label` is {reduce_zero_label}'
    if reduce_zero_label:
        # avoid using underflow conversion
        gt_semantic_seg[gt_semantic_seg == 0] = 255
        gt_semantic_seg = gt_semantic_seg - 1
        gt_semantic_seg[gt_semantic_seg == 254] = 255
    # # modify if custom classes
    # if results.get('label_map', None) is not None:
    #     # Add deep copy to solve bug of repeatedly
    #     # replace `gt_semantic_seg`, which is reported in
    #     # https://github.com/open-mmlab/mmsegmentation/pull/1445/
    #     gt_semantic_seg_copy = gt_semantic_seg.copy()
    #     for old_id, new_id in results['label_map'].items():
    #         gt_semantic_seg[gt_semantic_seg_copy == old_id] = new_id
    # results['gt_seg_map'] = gt_semantic_seg
    # results['seg_fields'].append('gt_seg_map')
    return gt_semantic_seg

def build_dataset(set_id, transform, data_root, mode='test', n_shot=None, split="all", bongard_anno=False, num_classes=None):
    if set_id == 'I':
        # ImageNet validation set
        # testdir = os.path.join(os.path.join(data_root, ID_to_DIRNAME[set_id]), 'val')
        testdir = os.path.join(os.path.join(ID_to_DIRNAME[set_id]), 'val')
        testset = datasets.ImageFolder(testdir, transform=transform)

    elif set_id in ['A', 'K', 'R', 'V']:
        if num_classes != '' and num_classes is not None:
            testdir = os.path.join(data_root, f'imagenet_v2_{num_classes}')
        else:
            testdir = os.path.join(data_root, ID_to_DIRNAME[set_id])
        testset = datasets.ImageFolder(testdir, transform=transform)

    elif set_id in list(path_dict.keys()) + ['aircraft', 'aircraft_sub']:
        if mode == 'train' and n_shot:
            testset = build_fewshot_dataset(set_id, os.path.join(data_root, ID_to_DIRNAME[set_id.lower()]), transform, mode=mode, n_shot=n_shot)
        else:
            testset = build_fewshot_dataset(set_id, os.path.join(data_root, ID_to_DIRNAME[set_id.lower()]), transform, mode=mode)
    else:
        raise NotImplementedError
        
    return testset

class build_dataset_remote(Dataset):
    """ remote dataset """
    def __init__(self, set_id, transform, data_root, mode='test', n_shot=None, split="all", bongard_anno=False, num_classes=None, args=None):
        self.set_id = Remote_ID_to_DIRNAME[set_id]    # 'LoveDA'
        self.resize = dict(type='Resize', scale=(args.resolution, args.resolution))
        self.transform = transform
        self.path = os.path.join(data_root,self.set_id)   
        self.mode = mode     

        if self.set_id == 'UDD5':
            self.img_path = os.path.join(self.path, 'val/src')
            print(self.img_path)
            self.ann_path = os.path.join(self.path, 'val/gt')
            print(self.ann_path)
        elif self.set_id == 'UAVid':
            self.img_path = os.path.join(self.path, 'img_dir/test')
            print(self.img_path)
            self.ann_path = os.path.join(self.path, 'ann_dir/test')
            print(self.ann_path)
        elif self.set_id == 'VDD':
            self.img_path = os.path.join(self.path, 'test/src')
            print( self.img_path)
            self.ann_path = os.path.join(self.path, 'test/gt')
            print(self.ann_path)
        elif self.set_id == 'WHU-BD':
            self.img_path = os.path.join(self.path, 'val/image')
            print(self.img_path)
            self.ann_path = os.path.join(self.path, 'val/label_cvt')
            print(self.ann_path)
        elif self.set_id == 'WHU_Sat':
            self.img_path = os.path.join(self.path, 'Satellite_dataset/cropped/test/image')
            print(self.img_path)
            self.ann_path = os.path.join(self.path, 'Satellite_dataset/cropped/test/label_cvt')
            print(self.ann_path)
        elif self.set_id == 'Inria':
            self.img_path = os.path.join(self.path, 'img_dir/split_test')
            print(self.img_path)
            self.ann_path = os.path.join(self.path, 'ann_dir/split_test')
            print(self.ann_path)
        elif self.set_id == 'xBD':
            self.img_path = os.path.join(self.path, 'test/images_pre')
            print(self.img_path)
            self.ann_path = os.path.join(self.path, 'test/targets_cvt_pre')
            print(self.ann_path)
        elif self.set_id == 'CHN6-CUG':
            self.img_path = os.path.join(self.path, 'val/image_cvt')
            print(self.img_path)
            self.ann_path = os.path.join(self.path, 'val/label_cvt')
            print(self.ann_path)
        elif self.set_id == 'DeepGlobe':
            self.img_path = os.path.join(self.path, 'image_cvt')
            print(self.img_path)
            self.ann_path = os.path.join(self.path, 'label_cvt')
            print( self.ann_path)
        elif set_id == 'massachusetts':
            self.img_path = os.path.join(self.path, 'Massachusetts_test_49/img')
            print(self.img_path)
            self.ann_path = os.path.join(self.path, 'Massachusetts_test_49/label_cvt')
            print(self.ann_path)
        elif set_id == 'spacenet':
            self.img_path = os.path.join(self.path, 'SpaceNet_test_567/img')
            print(self.img_path)
            self.ann_path = os.path.join(self.path, 'SpaceNet_test_567/label_cvt')
            print(self.ann_path)
        elif self.set_id == 'WBS-SI':
            self.img_path = os.path.join(self.path, 'Images')
            print(self.img_path)
            self.ann_path = os.path.join(self.path, 'Masks_cvt')
            print(self.ann_path)
        else:
            self.img_path = os.path.join(self.path, 'img_dir/val')
            print(self.img_path)
            self.ann_path = os.path.join(self.path, 'ann_dir/val')
            print(self.ann_path)

        IMG_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp')
        self.image_list = []
        self.label_list = []

        for fname in sorted(os.listdir(self.img_path)):
            if fname.lower().endswith(IMG_EXTENSIONS):
                self.image_list.append(os.path.join(self.img_path, fname))
                self.label_list.append(os.path.join(self.ann_path, fname))
        
        print('yt test image number:', len(self.image_list))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_path = self.image_list[idx]
        image_name = os.path.basename(image_path).split('.')[0] 
        image = Image.open(image_path).convert('RGB')
        ori_shape = image.size

        image_np = np.array(image)
        results = dict(img=image_np)
        resize_transform = Compose([self.resize])
        results = resize_transform(results)
        image = results['img']

        if self.transform:
            image = self.transform(image)

        if self.set_id == 'iSAID':
            mask = torch.tensor(load_seg_map(self.label_list[idx].split('.')[0]+'_instance_color_RGB.png',reduce_zero_label=False)).long()
        elif self.set_id == 'UDD5':
            mask = torch.tensor(load_seg_map(self.label_list[idx].split('.')[0]+'.png',reduce_zero_label=False)).long()
        elif self.set_id == 'UAVid':
            mask = torch.tensor(load_seg_map(self.label_list[idx].split('.')[0]+'.png',reduce_zero_label=False)).long()
        elif self.set_id == 'VDD':
            mask = torch.tensor(load_seg_map(self.label_list[idx].split('.')[0]+'.png',reduce_zero_label=False)).long()
        elif self.set_id in ['CHN6-CUG', 'DeepGlobe', 'GlobalRoadSet_Val']:
            mask = torch.tensor(load_seg_map(self.label_list[idx].split('.')[0]+'.png',reduce_zero_label=False)).long()
        elif self.set_id in ['LoveDA', 'potsdam', 'vaihingen']:
            mask = torch.tensor(load_seg_map(self.label_list[idx])).long()
        else:
            mask = torch.tensor(load_seg_map(self.label_list[idx],reduce_zero_label=False)).long()
        
        return image_name, image, ori_shape, mask

    
    # def get_affine_matrices(self, H, W, batch_size=4, device='cpu'):

    #     def build_affine_rotate_deg(deg):
    #         theta = deg * math.pi / 180.0
    #         return torch.tensor([
    #             [math.cos(theta), -math.sin(theta), 0.0],
    #             [math.sin(theta),  math.cos(theta), 0.0]
    #         ], dtype=torch.float32, device=device)

    #     def build_affine_flip(flip_h=False, flip_v=False):
    #         sx = -1.0 if flip_h else 1.0
    #         sy = -1.0 if flip_v else 1.0
    #         return torch.tensor([
    #             [sx, 0.0, 0.0],
    #             [0.0, sy, 0.0]
    #         ], dtype=torch.float32, device=device)

    #     # 定义6种基本变换
    #     base_transforms = [
    #         lambda: build_affine_rotate_deg(0),     # 原图
    #         lambda: build_affine_rotate_deg(90),
    #         lambda: build_affine_rotate_deg(180),
    #         lambda: build_affine_rotate_deg(270),
    #         lambda: build_affine_flip(flip_h=True),  # 水平翻转
    #         lambda: build_affine_flip(flip_v=True),  # 垂直翻转
    #     ]

    #     indices = torch.arange(batch_size) % 6
    #     matrices = torch.stack([base_transforms[i]() for i in indices])  # [B, 2, 3]
    #     return matrices


    # def apply_batch_affine(self, image, affine_matrices):
    #     if image.dim() == 3:
    #         image = image.unsqueeze(0)  # [C,H,W] -> [1,C,H,W]

    #     B = affine_matrices.shape[0]
    #     if image.size(0) == 1 and B > 1:
    #         image = image.expand(B, -1, -1, -1)

    #     # grid and sampling
    #     grid = F.affine_grid(
    #         affine_matrices, image.size(), align_corners=True
    #     )
    #     return F.grid_sample(
    #         image, grid, mode='bilinear', padding_mode='border', align_corners=True
    #     )

    # def align_logits(self, logit_patch, transform_matrices):
    #         """
    #         将不同变换的 logit 对齐回相同的空间位置（保持 logit 的尺寸不变）
            
    #         参数:
    #             logit_patch: 变换后的 logit, 形状为 [batch_img, cls, h, w]
    #             transform_matrices: 原始变换矩阵, 形状为 [batch_img, 2, 3]
            
    #         返回:
    #             对齐后的 logit, 形状仍为 [batch_img, cls, h, w]
    #         """
    #         batch_size, num_classes, h, w = logit_patch.shape
            
    #         # 1. 计算每个变换矩阵的逆矩阵
    #         # 首先将2x3矩阵扩展为3x3齐次坐标矩阵
    #         eye = torch.eye(3, device=transform_matrices.device).unsqueeze(0)  # [1,3,3]
    #         eye = eye.expand(batch_size, -1, -1)  # [B,3,3]
            
    #         # 填充第三行 [0,0,1]
    #         transform_matrices_homo = torch.cat([
    #             transform_matrices,
    #             torch.tensor([0, 0, 1], device=transform_matrices.device)
    #                 .reshape(1,1,3).expand(batch_size, -1, -1)
    #         ], dim=1)  # [B,3,3]
            
    #         # 计算逆矩阵
    #         inv_matrices_homo = torch.inverse(transform_matrices_homo)
    #         inv_matrices = inv_matrices_homo[:, :2, :]  # 取前两行 [B,2,3]
            
    #         # 2. 创建目标网格（保持 logit 的尺寸）
    #         grid = F.affine_grid(
    #             inv_matrices, 
    #             logit_patch.size(),  # 保持输出尺寸与输入相同
    #             align_corners=True
    #         )
            
    #         # 3. 应用逆变换
    #         aligned_logit = F.grid_sample(
    #             logit_patch,
    #             grid,
    #             mode='bilinear',
    #             padding_mode='border',
    #             align_corners=True
    #         )
            
    #         return aligned_logit

def get_preaugment():
    return transforms.Compose([
            transforms.RandomResizedCrop(448),
            transforms.RandomHorizontalFlip(),
        ])    

def augmix(image, preprocess, aug_list, severity=1):
    preaugment = get_preaugment()
    x_orig = preaugment(image)
    x_processed = preprocess(x_orig)
    if len(aug_list) == 0:
        return x_processed
    w = np.float32(np.random.dirichlet([1.0, 1.0, 1.0]))
    m = np.float32(np.random.beta(1.0, 1.0))

    mix = torch.zeros_like(x_processed)
    for i in range(3):
        x_aug = x_orig.copy()
        for _ in range(np.random.randint(1, 4)):
            x_aug = np.random.choice(aug_list)(x_aug, severity)
        mix += w[i] * preprocess(x_aug)
    mix = m * x_processed + (1 - m) * mix
    return mix


class AugMixAugmenter(object):
    def __init__(self, base_transform, preprocess, n_views=2, augmix=False, 
                    severity=1):
        self.base_transform = base_transform
        self.preprocess = preprocess
        self.n_views = n_views
        if augmix:
            self.aug_list = augmentations.augmentations
        else:
            self.aug_list = []
        self.severity = severity
        
    def __call__(self, x):
        image = self.preprocess(self.base_transform(x))
        views = [augmix(x, self.preprocess, self.aug_list, self.severity) for _ in range(self.n_views)]
        return [image] + views

class SegmentationAugmenter(object):
    def __init__(self, base_transform, preprocess, n_views=1):
        self.base_transform = base_transform
        self.preprocess = preprocess
        self.n_views = n_views

    def build_random_augment(self):
        return transforms.Compose([
            transforms.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
                hue=0.1
            ),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            transforms.RandomAutocontrast(p=0.5),
            transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
        ])

    def __call__(self, img):
        base_img = self.base_transform(img)
        image = self.preprocess(base_img)

        views = []
        for _ in range(self.n_views):
            aug = self.build_random_augment()
            aug_img = aug(base_img)  
            aug_img = self.preprocess(aug_img)
            views.append(aug_img)

        return [image] + views

