import argparse
import os
import torch
import torch.nn.functional as F
import pickle

from tqdm import tqdm

import sys
sys.path.append(os.getcwd())

from model import load, tokenize, DOWNLOAD_ROOT
from model.text_encoders import TextEncoderWithPrompt
from data.imagenet_prompts_clean import imagenet_classes, imagenet_templates
from data.cls_to_names_remote import (OpenEarthMap_classes,potsdam_classes,LoveDA_classes,iSAID_classes,uavid_classes,udd5_classes,vaihingen_classes,vdd_classes,road_classes,water_classes,building_classes)   

CLASSES_DICT = {
    "OpenEarthMap":OpenEarthMap_classes,
    "potsdam":potsdam_classes,
    "LoveDA":LoveDA_classes,
    "iSAID":iSAID_classes,
    "uavid":uavid_classes,
    "udd5":udd5_classes,
    "vaihingen":vaihingen_classes,
    "vdd":vdd_classes,
    "whu":building_classes,
    "Inria":building_classes,
    "xBD":building_classes,
    "CHN6-CUG":road_classes,
    "DeepGlobe":road_classes,
    "Massachusetts":road_classes,
    "SpaceNet":road_classes,
    "WBS-SI":water_classes,
}   

device='cuda'

n_ctx = 4

coop_path = 'checkpoints/to_gdrive/vit_b16_ep50_16shots/nctx4_cscFalse_ctpend/seed1/prompt_learner/model.pth.tar-50'
ctx = torch.load(coop_path)['state_dict']['ctx'].unsqueeze(0)


def main(args):

    clip, _, _ = load(args.arch, device=device, download_root=DOWNLOAD_ROOT)
    dtype = clip.visual.conv1.weight.dtype
    text_encoder_w_prompt = TextEncoderWithPrompt(clip)

    if args.x_templates:
        dir_class_embeds = 'remote_class_embeds_w_imagenet_templates' 
    elif args.coop:
        dir_class_embeds = 'remote_coop_embeds'  
    else:
        dir_class_embeds = 'remote_class_embeds'  
    
    dir_class_embeds = os.path.join(args.arch.replace('/', '-').lower() + "_embeds", dir_class_embeds)

    os.makedirs(dir_class_embeds, exist_ok=True)

    print(f"Saving class embeds to {dir_class_embeds}")

    DATASETS = ["OpenEarthMap","potsdam","LoveDA", "iSAID","uavid","udd5","vaihingen","vdd","whu","Inria","xBD","CHN6-CUG","DeepGlobe","Massachusetts","SpaceNet","WBS-SI"]   

    for dataset in DATASETS:
        print(f"Dataset: {dataset}")
        class_embeds_path = os.path.join(dir_class_embeds, f'{dataset.lower()}.pkl')

        classes_lst = CLASSES_DICT[dataset]
        classes_lst = [name.replace("_", " ") for name in classes_lst]

        class_embeds = {}
        for classname in tqdm(classes_lst, total=len(classes_lst)):
            assert "_" not in classname

            class_names = classname.split(',')  
            class_names = [item.replace('\n', '') for item in class_names]
            class_features = []    

            for classname_i in class_names:
            
                if args.x_templates:
                    prompts = [template.format(classname_i) for template in imagenet_templates]
                else:
                    prompts = [f'a photo of a {classname_i}.']  

                tokenized_prompts = tokenize(prompts).to(device)
                if args.coop:
                    with torch.no_grad():
                        embedding = clip.token_embedding(tokenized_prompts).type(dtype)

                    prefix = embedding[:, :1, :]
                    suffix = embedding[:, 1 + n_ctx :, :]  # CLS, EOS

                    prompts = torch.cat(
                        [
                            prefix,  # (n_cls, 1, dim)
                            ctx,     # (n_cls, n_ctx, dim)
                            suffix,  # (n_cls, *, dim)
                        ],
                        dim=-2,
                    )

                    embeds = text_encoder_w_prompt(prompts, tokenized_prompts)
                else:
                    with torch.no_grad():
                        embeds = clip.encode_text(tokenized_prompts).cpu()
                
                embeds = F.normalize(embeds, dim=-1)
                print('embeds.shape:', embeds.shape)   #embeds.shape: torch.Size([1, 512])
                class_features.append(embeds.unsqueeze(0))  

            class_embeds_ = torch.cat(class_features, dim=0)   
            class_embeds[classname] = class_embeds_.mean(dim=0)

        print(f"Dumping class embeds to {class_embeds_path}")
        with open(class_embeds_path, 'wb') as f:
            pickle.dump(class_embeds, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type=str, default='ViT-B/16', choices=['ViT-B/16', 'RN50'])
    parser.add_argument('--x_templates', action='store_true', help='whether to use imagenet templates')
    parser.add_argument('--coop', action='store_true', help='whether to use coop prefix')

    args = parser.parse_args()

    main(args)
