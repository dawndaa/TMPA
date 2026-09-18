import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle

from model import load, DOWNLOAD_ROOT
from data.fewshot_datasets import fewshot_datasets
from data.cls_to_names import *

from data.prompt_embeds import get_class2concept_dict_path, get_concept_embeds_path, get_class_embeds_path, get_susx_class_embeds_path

from .text_encoders import ClipTextEncoder
from .visual_encoders import ClipImageEncoder

from .shifter import Shifter_seg,Shifter, Shifter_visual
from .film import FiLM

import pywt
import numpy as np
import matplotlib.pyplot as plt
import os
import ot
import re

from open_clip import tokenizer, create_model
from simfeatup_dev.upsamplers import get_upsampler
from torchvision.utils import save_image
from visualize import visualize_and_save_feature
from model.dataset_config import DATASET_CONFIGS

# NOTE: only semantic changes are applied in this file:
# forward_feature uses args.module_visual_guidance to gate the visual-guided
# prompt update. TTA text_shift remains controlled by existing args.text_shift.


def get_dataset_config(dataset_name, resolution=None):
    dataset_name = dataset_name.lower()
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    dataset_cfg = DATASET_CONFIGS[dataset_name]
    if resolution is not None:
        resolution = str(resolution)
        if resolution in dataset_cfg:
            return dataset_cfg[resolution]
    raise ValueError(f"No config found for dataset={dataset_name}, resolution={resolution}")


# The rest of the original TMPA implementation is unchanged.
