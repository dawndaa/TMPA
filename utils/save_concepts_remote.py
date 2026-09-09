import argparse
import os
import json
import torch
import torch.nn.functional as F
import pickle
import numpy as np

import re

from tqdm import tqdm

import sys
sys.path.append(os.getcwd())
from model import load, tokenize
from data.imagenet_prompts_clean import imagenet_classes, imagenet_templates

# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from open_clip import tokenizer, create_model
from data.cls_to_names_remote import (
    OpenEarthMap_classes,
    potsdam_classes,
    LoveDA_classes,
    iSAID_classes,
    uavid_classes,
    udd5_classes,
    vaihingen_classes,
    vdd_classes,
    road_classes,
    water_classes,
    building_classes
)

CLASSES_DICT = {
    "OpenEarthMap": OpenEarthMap_classes,
    "potsdam": potsdam_classes,
    "LoveDA": LoveDA_classes,
    "iSAID": iSAID_classes,
    "uavid": uavid_classes,
    "udd5": udd5_classes,
    "vaihingen": vaihingen_classes,
    "vdd": vdd_classes,
    "whu_aerial": building_classes,
    "whu_sat": building_classes,
    "inria": building_classes,
    "xbd": building_classes,
    "CHN6-CUG": road_classes,
    "DeepGlobe": road_classes,
    "Massachusetts": road_classes,
    "SpaceNet": road_classes,
    "WBS_SI": water_classes,
}

openai_imagenet_template = [
    lambda c: f'a bad photo of a {c}.',
    lambda c: f'a photo of many {c}.',
    lambda c: f'a sculpture of a {c}.',
    lambda c: f'a photo of the hard to see {c}.',
    lambda c: f'a low resolution photo of the {c}.',
    lambda c: f'a rendering of a {c}.',
    lambda c: f'graffiti of a {c}.',
    lambda c: f'a bad photo of the {c}.',
    lambda c: f'a cropped photo of the {c}.',
    lambda c: f'a tattoo of a {c}.',
    lambda c: f'the embroidered {c}.',
    lambda c: f'a photo of a hard to see {c}.',
    lambda c: f'a bright photo of a {c}.',
    lambda c: f'a photo of a clean {c}.',
    lambda c: f'a photo of a dirty {c}.',
    lambda c: f'a dark photo of the {c}.',
    lambda c: f'a drawing of a {c}.',
    lambda c: f'a photo of my {c}.',
    lambda c: f'the plastic {c}.',
    lambda c: f'a photo of the cool {c}.',
    lambda c: f'a close-up photo of a {c}.',
    lambda c: f'a black and white photo of the {c}.',
    lambda c: f'a painting of the {c}.',
    lambda c: f'a painting of a {c}.',
    lambda c: f'a pixelated photo of the {c}.',
    lambda c: f'a sculpture of the {c}.',
    lambda c: f'a bright photo of the {c}.',
    lambda c: f'a cropped photo of a {c}.',
    lambda c: f'a plastic {c}.',
    lambda c: f'a photo of the dirty {c}.',
    lambda c: f'a jpeg corrupted photo of a {c}.',
    lambda c: f'a blurry photo of the {c}.',
    lambda c: f'a photo of the {c}.',
    lambda c: f'a good photo of the {c}.',
    lambda c: f'a rendering of the {c}.',
    lambda c: f'a {c} in a video game.',
    lambda c: f'a photo of one {c}.',
    lambda c: f'a doodle of a {c}.',
    lambda c: f'a close-up photo of the {c}.',
    lambda c: f'a photo of a {c}.',
    lambda c: f'the origami {c}.',
    lambda c: f'the {c} in a video game.',
    lambda c: f'a sketch of a {c}.',
    lambda c: f'a doodle of the {c}.',
    lambda c: f'a origami {c}.',
    lambda c: f'a low resolution photo of a {c}.',
    lambda c: f'the toy {c}.',
    lambda c: f'a rendition of the {c}.',
    lambda c: f'a photo of the clean {c}.',
    lambda c: f'a photo of a large {c}.',
    lambda c: f'a rendition of a {c}.',
    lambda c: f'a photo of a nice {c}.',
    lambda c: f'a photo of a weird {c}.',
    lambda c: f'a blurry photo of a {c}.',
    lambda c: f'a cartoon {c}.',
    lambda c: f'art of a {c}.',
    lambda c: f'a sketch of the {c}.',
    lambda c: f'a embroidered {c}.',
    lambda c: f'a pixelated photo of a {c}.',
    lambda c: f'itap of the {c}.',
    lambda c: f'a jpeg corrupted photo of the {c}.',
    lambda c: f'a good photo of a {c}.',
    lambda c: f'a plushie {c}.',
    lambda c: f'a photo of the nice {c}.',
    lambda c: f'a photo of the small {c}.',
    lambda c: f'a photo of the weird {c}.',
    lambda c: f'the cartoon {c}.',
    lambda c: f'art of the {c}.',
    lambda c: f'a drawing of the {c}.',
    lambda c: f'a photo of the large {c}.',
    lambda c: f'a black and white photo of a {c}.',
    lambda c: f'the plushie {c}.',
    lambda c: f'a dark photo of a {c}.',
    lambda c: f'itap of a {c}.',
    lambda c: f'graffiti of the {c}.',
    lambda c: f'a toy {c}.',
    lambda c: f'itap of my {c}.',
    lambda c: f'a photo of a cool {c}.',
    lambda c: f'a photo of a small {c}.',
    lambda c: f'a tattoo of the {c}.',
]

device='cuda'
DOWNLOAD_ROOT='checkpoints/clip'

def clean_string(expression):
    return re.sub(r"([.,'!?\"()*#:;])", '', expression.lower()).replace('-', ' ').replace('/', ' ')

def make_descriptor_sentence(descriptor):
# Code from https://github.com/sachit-menon/classify_by_description_release/blob/master/descriptor_strings.py#L43
    if descriptor.startswith('a') or descriptor.startswith('an'):
        return f"which is {descriptor}"
    elif descriptor.startswith('has') or descriptor.startswith('often') or descriptor.startswith('typically') or descriptor.startswith('may') or descriptor.startswith('can'):
        return f"which {descriptor}"
    elif descriptor.startswith('used'):
        return f"which is {descriptor}"
    else:
        return f"which has {descriptor}"


def save_concepts(args):
    # clip, _, _ = load(args.arch, device=device, download_root=DOWNLOAD_ROOT)
    net = create_model('ViT-B/16', pretrained='openai', precision='fp16').to(device)

    if args.no_cond:
        suffix = "_no_cond"
    elif args.x_templates:
        suffix = "_x_templates"
    else:
        suffix = ""

    gpt4_concepts_embeds_dir = os.path.join(args.arch.replace('/', '-').lower() + "_embeds", args.gpt4_concepts_embeds_dir + suffix)

    os.makedirs(gpt4_concepts_embeds_dir, exist_ok=True)

    concepts_json = args.concepts_json
    dataset = args.dataset

    concept_embeds_path = os.path.join(gpt4_concepts_embeds_dir, f'{dataset.lower()}.pkl')

    print(f"Saving concept embeds to {concept_embeds_path}")

    if args.save_concept_dict:
        concept_dict_dir = args.concept_dict_dir + suffix
        os.makedirs(concept_dict_dir, exist_ok=True)
        concept_dict_path = os.path.join(concept_dict_dir, f'{dataset}.json')

    print(f"Loading concepts from {concepts_json}")
    with open(concepts_json, 'r') as f:
        concepts_dict = json.load(f)
    
    gpt4_classes = list(concepts_dict.keys())
    tpt_classes = CLASSES_DICT[dataset]
    
    concept_embeds = {}
    concept_dict_all = {}

    len_concepts = np.array([len(concepts_dict[classname_gpt4]) for classname_gpt4 in gpt4_classes])
    empty_indices = np.where(len_concepts == 0)[0]
    empty_classnames = [gpt4_classes[i] for i in empty_indices]

    assert len(empty_classnames) == 0, f"Empty classnames: {empty_classnames}"

    tpt_classes = [name.replace("_", " ") for name in tpt_classes]

    for classname_tpt, classname_gpt4 in tqdm(zip(tpt_classes, gpt4_classes), total=len(tpt_classes)):
        assert "_" not in classname_tpt
        
        concepts = concepts_dict[classname_gpt4]

        assert len(concepts) > 0, f"Empty concepts for class {classname_gpt4} in dataset {dataset}"
        
        if args.no_cond:
            prompts = concepts
        elif args.x_templates:
            prompts = [t.format(f"{classname_tpt}, " + make_descriptor_sentence(c)) for c in concepts for t in imagenet_templates]
        else:
            prompts = [c for c in concepts]   
 
        class_names = [clean_string(item.replace('\n', '')) for item in prompts]
        query_features = []
        with torch.no_grad():
            for qw in class_names:
                query = tokenizer.tokenize([temp(qw) for temp in openai_imagenet_template]).to(device)
                feature = net.encode_text(query)
                feature /= feature.norm(dim=-1, keepdim=True)
                feature = feature.mean(dim=0)
                feature /= feature.norm()
                query_features.append(feature.unsqueeze(0))

        embeds = torch.cat(query_features, dim=0) 
        concept_embeds[classname_tpt] = embeds

        concept_dict_all[classname_tpt] = prompts
    
    if args.save_concept_dict:
        print(f"Dumping concept dict to {concept_dict_path}")
        with open(concept_dict_path, 'w') as f:
            json.dump(concept_dict_all, f, indent=4)
    
    print(f"Dumping concept embeds to {concept_embeds_path}")
    with open(concept_embeds_path, 'wb') as f:
        pickle.dump(concept_embeds, f)

def process_json_folder(args):
    json_dir = args.batch_json_dir
    json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]

    if not json_files:
        print(f"[Error] No JSON files found in {json_dir}")
        return

    print(f"[Info] Found {len(json_files)} JSON files in {json_dir}")

    for json_file in json_files:
        dataset_name = os.path.splitext(json_file)[0]
        if dataset_name not in CLASSES_DICT:
            print(f"[Skip] {json_file}: dataset not recognized.")
            continue

        print(f"\n=== Processing dataset: {dataset_name} ===")
        args.concepts_json = os.path.join(json_dir, json_file)
        args.dataset = dataset_name
        save_concepts(args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--concepts_json', type=str, default='concepts/imagenet-gpt4-full-v4.json')
    parser.add_argument('--dataset', type=str, default='LoveDA')
    parser.add_argument('--batch_json_dir', type=str, default=None)   
    parser.add_argument('--no_cond', action="store_true")
    parser.add_argument('--x_templates', action="store_true")
    parser.add_argument('--arch', type=str, default='ViT-B/16', choices=['ViT-B/16', 'RN50'])
    parser.add_argument('--gpt4_concepts_embeds_dir', type=str, default='remote_concept_embeds_gpt4')
    parser.add_argument('--concept_dict_dir', type=str, default='concept_dict_llm')
    parser.add_argument('--save_concept_dict', action='store_true')

    args = parser.parse_args()

    if args.batch_json_dir:
        process_json_folder(args)
    else:
        save_concepts(args)
    