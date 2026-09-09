import time
import os
from copy import deepcopy

from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn  
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms

import torchvision.transforms as T   
import pandas as pd   
import numpy as np   

import matplotlib.pyplot as plt  
import matplotlib.colors as mcolors   
import os   
from PIL import Image   
from collections import OrderedDict
from prettytable import PrettyTable
from mmengine.logging import MMLogger, print_log

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

from model.learnable_shift import get_shift_model
from data.imagenet_prompts_clean import imagenet_classes
from data.datautils import AugMixAugmenter, build_dataset_remote, SegmentationAugmenter
from utils.tools import Summary, AverageMeter, ProgressMeter, accuracy, load_model_weight, set_random_seed
from data.cls_to_names import *
from data.fewshot_datasets import fewshot_datasets
from data.imagenet_variants import (
    thousand_k_to_200, 
    imagenet_a_mask, 
    imagenet_r_mask, 
    imagenet_v_mask,
)
from data.cls_to_names_remote import (OpenEarthMap_classes,potsdam_classes,LoveDA_classes,iSAID_classes,uavid_classes,udd5_classes,vaihingen_classes,vdd_classes,road_classes,water_classes,building_classes)  

from utils.utils import (
                         AverageMeter, Summary, 
                         intersectionAndUnionGPU, intersect_and_union, total_area_to_metrics)  

CLASSES_DICT = {
    "openearthmap": OpenEarthMap_classes,
    "loveda": LoveDA_classes,
    "isaid": iSAID_classes,
    "potsdam": potsdam_classes,
    "uavid":uavid_classes,
    "udd5":udd5_classes,
    "vaihingen": vaihingen_classes,
    "vdd":vdd_classes,
    "whu_aerial":building_classes,
    "whu_sat":building_classes,
    "inria":building_classes,
    "xbd":building_classes,
    "chn6-cug":road_classes,
    "deepglobe":road_classes,
    "massachusetts":road_classes,
    "spacenet":road_classes,
    "wbs_si":water_classes,
}  

from torch.utils.data.distributed import DistributedSampler
from run_utils import select_confident_samples,select_confident_samples_seg, avg_entropy, model_names, IMAGENET_VARIANTS, log_results, avg_entropy_seg, loss_prompt_logit,loss_prompt_entropy
from args import parse_args

from color import OpenEarthMap_PALETTE, potsdam_PALETTE, LoveDA_PALETTE, iSAID_PALETTE, uavid_PALETTE, udd5_PALETTE, vaihingen_PALETTE, vdd_PALETTE, road_PALETTE, water_PALETTE, building_PALETTE


class ResizeKeepRatio(object):
    def __init__(self, long_size=448):
        self.long_size = long_size

    def __call__(self, img):
        w, h = img.size
        if w >= h:
            new_w, new_h = self.long_size, int(h * self.long_size / w)
        else:
            new_h, new_w = self.long_size, int(w * self.long_size / h)
        return img.resize((new_w, new_h), Image.BILINEAR)


def compute_iou(pred_label, label):
    intersection = ((label * pred_label) > 0).sum()
    union = ((label + pred_label) > 0).sum()
    return intersection / union

def setup_distributed(args):
    """Initialize distributed training environment."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ['RANK'])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.local_rank = int(os.environ['LOCAL_RANK'])
    else:
        print('Not using distributed mode')
        args.distributed = False
        args.rank = 0
        return

    args.distributed = True
    torch.cuda.set_device(args.local_rank)
    
    torch.distributed.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=args.world_size,
        rank=args.rank
    )
    torch.distributed.barrier()
    print(f"Distributed training initialized (rank {args.rank}, local rank {args.local_rank})")

def run_shift_iter(model, inputs,ori_shape, args):
    """Run one iteration of shift model."""
    output = model(inputs, ori_shape, args)    
    preds = F.softmax(output[0], dim=-1)   
    pred = torch.argmax(preds).item()

    output, _ = select_confident_samples_seg(output, args.selection_p)   
    loss = avg_entropy(output)

    return loss, pred


def run_shift_iter_seg(model, inputs,ori_shape, image_name, args):
    """Run one iteration of shift model."""

    if args.loss_prompt == 'True': 
        mask_pred, seg_pred_logits,all_seg_logits = model(inputs, ori_shape, image_name, args) 
    else:
        mask_pred, seg_pred_logits = model(inputs, ori_shape, image_name, args) 

    loss_entropy = avg_entropy_seg(seg_pred_logits, args.tps_entropy_scale)

    if args.loss_prompt == 'True': 
        loss_prompt = loss_prompt_entropy(all_seg_logits, seg_pred_logits, logit_scale=args.prompt_logit_scale, args=args)
        total_loss = (loss_prompt + loss_entropy)/2

    else:
        total_loss = loss_entropy

    return total_loss, mask_pred

def postprocess_result(seg_logits, logit_scale = 50):
    seg_logits_p = seg_logits * logit_scale  
    seg_logits_p = seg_logits_p.softmax(2) 

    return seg_logits_p


def test_time_tuning(image_name, model, inputs, ori_shape, optimizer, scaler, args):    
    """Perform test-time adaptation."""
    for j in range(args.tta_steps):
        with torch.cuda.amp.autocast():
            loss, pred = run_shift_iter_seg(model, inputs, ori_shape, image_name, args) 
        if loss.requires_grad and any(p.requires_grad for p in model.parameters()):
            optimizer.zero_grad()
            scaler.scale(loss).backward(retain_graph=True)
            scaler.step(optimizer)
            scaler.update()
    return pred

def main(args):
    """Main function for distributed training."""
    setup_distributed(args)
    main_worker(args.local_rank, args)

def main_worker(local_rank, args):
    """Main worker function for each process."""

    num_workers = min(4, os.cpu_count() // args.world_size)  
    loader_args = {
        'batch_size': args.batch_size,
        'num_workers': num_workers,
        'pin_memory': True,
        'persistent_workers': num_workers > 0,
        'drop_last': False  
    }

    set_random_seed(args.seed + args.rank)
    device = torch.device(f"cuda:{local_rank}")   

    # Only print on main process
    if args.rank == 0:
        print(f"Use GPU: {local_rank} for training")

    # Iterate through evaluation datasets
    datasets = args.test_sets.split("/")
    results = {}

    for set_id in datasets:
        args.test_sets = set_id

        # Normalization stats from clip.load()
        normalize = transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711]
        ) 
        
        args.img_aug = False    

        if args.img_aug:

            base_transform = transforms.Resize((args.resolution, args.resolution), interpolation=BICUBIC)
            preprocess = transforms.Compose([
                transforms.ToTensor(),
                normalize
            ])

            data_transform = SegmentationAugmenter(
                base_transform=base_transform,
                preprocess=preprocess,
                n_views=args.batch_size-1  
            )
        else:
            data_transform = transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])

        batchsize = 1

        if args.rank == 0:
            print(f"Evaluating: {set_id}")

        classnames = CLASSES_DICT[set_id]
        # print(classnames)
        
        # Load model
        model = get_shift_model(args, classnames)
        
        model = model.to(device)   
        for name, param in model.named_parameters():   
            if param.device != device:
                print(f"[Warning] Parameter {name} not on {device}, moving manually.")
                param.data = param.data.to(device)

        # Load pre-trained weights if specified
        if args.load is not None and args.rank == 0:
            print("Use pre-trained soft prompt (CoOp) as initialization")
            pretrained_ctx = torch.load(args.load)['state_dict']['ctx']
            assert pretrained_ctx.size()[0] == args.n_ctx
            with torch.no_grad():
                model.prompt_learner.ctx.copy_(pretrained_ctx)
                model.prompt_learner.ctx_init_state = pretrained_ctx

        # Freeze parameters that shouldn't be trained
        for name, param in model.named_parameters():
            if (args.tpt and "prompt_learner" not in name) and (args.scale and "scaler" not in name):
                param.requires_grad_(False)

        if args.rank == 0:
            print(f"=> Model created: visual backbone {args.arch}")

        # Initialize distributed data parallel
        if torch.cuda.is_available():
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=True
            )
        else:
            if args.rank == 0:
                print('Warning: using CPU, this will be slow')

        cudnn.benchmark = True

        val_dataset = build_dataset_remote(
            set_id = set_id, 
            transform = data_transform, 
            data_root = args.data, 
            mode=args.dataset_mode, 
            num_classes=args.num_classes,
            args = args
        )
        val_sampler = DistributedSampler(
            val_dataset,
            shuffle=False,
            drop_last=False
        )

        val_loader = torch.utils.data.DataLoader(
                val_dataset,
                sampler=val_sampler,
                **loader_args
            )
        
        if args.rank == 0:
            print(f"Number of test samples: {len(val_dataset)}")

        # Define optimizer
        optimizer, optim_state = None, None
        trainable_param = []

        if args.text_shift:  
        
            if hasattr(model, 'module'):
                trainable_param += list(model.module.text_shifter.parameters())
                if hasattr(model.module, 'text_shifter_visual'):
                    trainable_param += list(model.module.text_shifter_visual.parameters())
            else:
                trainable_param += list(model.text_shifter.parameters())
                if hasattr(model, 'text_shifter_visual'):
                    trainable_param += list(model.text_shifter_visual.parameters())

        if args.tta_steps > 0:
            m = model.module if hasattr(model, 'module') else model
            alpha_lr_factor = 1.0
            other_params = [p for p in trainable_param if p is not m.alpha]
            param_groups = [
                {'params': other_params, 'lr': args.lr}, 
                {'params': [m.alpha], 'lr': args.lr * alpha_lr_factor}
            ]
            optimizer = torch.optim.AdamW(param_groups)
            optim_state = deepcopy(optimizer.state_dict())

        # Setup automatic mixed-precision
        scaler = torch.cuda.amp.GradScaler(init_scale=1e3)

        if args.rank == 0:
            print('=> Using native Torch AMP. Training in mixed precision.')
            
        # Run evaluation
        results[set_id] = test_time_adapt_eval(val_loader, model, optimizer, optim_state, scaler,classnames, args)
        
        # Clean up
        del val_dataset, val_loader
        torch.cuda.empty_cache()

def test_time_adapt_eval(val_loader, model, optimizer, optim_state, scaler,classnames, args):
    """Evaluate model with test-time adaptation."""

    results = []   
    logger: MMLogger = MMLogger.get_current_instance()   


    model.eval()

    for i, (image_name, images, ori_shape, target) in tqdm(enumerate(val_loader), total=len(val_loader), disable=args.rank != 0):   
        # Move data to GPU
        if isinstance(images, list):
            for k in range(len(images)):
                images[k] = images[k].cuda(args.local_rank, non_blocking=True)
            image = images[0]
        else:
            if len(images.size()) > 4:
                assert images.size()[0] == 1
                images = images.squeeze(0)
            images = images.cuda(args.local_rank, non_blocking=True)
            image = images
        target = target.cuda(args.local_rank, non_blocking=True)

        # Reset tunable parameters to initial state
        if args.tta_steps > 0:
            with torch.no_grad():
                model.module.reset(args) if hasattr(model, 'module') else model.reset()

        if optim_state is not None:
            optimizer.load_state_dict(optim_state)

        # Perform test-time tuning if needed
        if args.tta_steps > 0:
            test_time_tuning(image_name[0], model, images, ori_shape, optimizer, scaler, args)


        # Forward pass
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                if args.loss_prompt == 'True':
                    mask_pred,visual_features,loss_prompt = model(image, ori_shape, image_name[0], args,test=True)  

                else:
                    mask_pred,visual_features = model(image, ori_shape, image_name[0], args,test=True)   
        

        target =target.to(mask_pred.device)   
        for mask_i, output_i in zip(target, mask_pred):
            results.append(intersect_and_union(
                output_i.int().contiguous().clone(), mask_i.int().contiguous(), args.num_classes, ignore_index=255
            ))
    results = tuple(zip(*results))
    assert len(results) == 4    

    total_area_intersect = sum(results[0]).clone().detach().to(dtype=torch.float64, device='cuda')
    total_area_union = sum(results[1]).clone().detach().to(dtype=torch.float64, device='cuda')
    total_area_pred_label = sum(results[2]).clone().detach().to(dtype=torch.float64, device='cuda')
    total_area_label = sum(results[3]).clone().detach().to(dtype=torch.float64, device='cuda')

    torch.distributed.all_reduce(total_area_intersect, op=torch.distributed.ReduceOp.SUM)
    torch.distributed.all_reduce(total_area_union, op=torch.distributed.ReduceOp.SUM)
    torch.distributed.all_reduce(total_area_pred_label, op=torch.distributed.ReduceOp.SUM)
    torch.distributed.all_reduce(total_area_label, op=torch.distributed.ReduceOp.SUM)

    ret_metrics = total_area_to_metrics(
    total_area_intersect.cpu().numpy(),
    total_area_union.cpu().numpy(),
    total_area_pred_label.cpu().numpy(),
    total_area_label.cpu().numpy())
    
    class_names = classnames
    print('class_names', class_names)

    # summary table
    ret_metrics_summary = OrderedDict({
        ret_metric: np.round(np.nanmean(ret_metric_value) * 100, 2)
        for ret_metric, ret_metric_value in ret_metrics.items()
    })
    metrics = dict()
    for key, val in ret_metrics_summary.items():
        if key == 'aAcc':
            metrics[key] = val
        else:
            metrics['m' + key] = val
    
    # each class table
    ret_metrics.pop('aAcc', None)
    ret_metrics_class = OrderedDict({
        ret_metric: np.round(ret_metric_value * 100, 2)
        for ret_metric, ret_metric_value in ret_metrics.items()
    })
    ret_metrics_class.update({'Class': class_names})
    ret_metrics_class.move_to_end('Class', last=False)
    class_table_data = PrettyTable()
    for key, val in ret_metrics_class.items():
        class_table_data.add_column(key, val)

    print_log('per class results:', logger)
    print_log('\n' + class_table_data.get_string(), logger=logger)
    print(f"aAcc: {metrics['aAcc']:.2f}, mIoU: {metrics['mIoU']:.2f}, mAcc: {metrics['mAcc']:.2f}")

    os.makedirs("save_result", exist_ok=True)
    result_file = os.path.join("save_result", args.save_result if hasattr(args, "save_result") else "results.txt")

    with open(result_file, "w") as f:
        f.write("Summary metrics:\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v:.2f}\n")

        f.write("\nPer class results:\n")
        f.write(class_table_data.get_string() + "\n")

    print(f"[INFO] Results saved to {result_file}")

    return None

if __name__ == '__main__':
    args = parse_args()
    main(args)