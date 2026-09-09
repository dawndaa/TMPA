import os
import torch
import numpy as np
from PIL import Image

def visualize_and_save_feature(feature_tensor, save_path, index=0):

    if feature_tensor.is_cuda:
        feature_tensor = feature_tensor.cpu()
    
    feat = feature_tensor[index]  # [C, H, W]

    feat_map = feat.mean(dim=0)  # [H, W]

    feat_map -= feat_map.min()
    feat_map /= (feat_map.max() + 1e-8)

    feat_map_img = (feat_map.numpy() * 255).astype(np.uint8)

    img = Image.fromarray(feat_map_img, mode='L')

    os.makedirs(save_path, exist_ok=True)

    save_file = os.path.join(save_path, f'feature_map_{index}.png')
    img.save(save_file)

    print(f'Saved feature visualization to {save_file}')
