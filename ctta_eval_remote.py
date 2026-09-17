"""Strict single-stream CTTA evaluation for TMPA remote-sensing datasets.

Keeps TMPA's remote datasets/model/adaptation, reuses DAF corruptions, and
changes only the temporal reset policy. One temporal stream runs per process.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
import torchvision.transforms as transforms
from tqdm import tqdm

from args import parse_args
from ctta.corruptions import expand_corruptions
from ctta.dataset import CorruptedRemoteDataset
from ctta.metrics import summarize_progress, summarize_results
from ctta.state import AdaptationStateController
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
    building_classes,
)
from model.learnable_shift import get_shift_model
from shift_classification_remote import test_time_tuning
from utils.tools import set_random_seed
from utils.utils import intersect_and_union


CLASSES_DICT = {
    'openearthmap': OpenEarthMap_classes,
    'loveda': LoveDA_classes,
    'isaid': iSAID_classes,
    'potsdam': potsdam_classes,
    'uavid': uavid_classes,
    'udd5': udd5_classes,
    'vaihingen': vaihingen_classes,
    'vdd': vdd_classes,
    'whu_aerial': building_classes,
    'whu_sat': building_classes,
    'inria': building_classes,
    'xbd': building_classes,
    'chn6-cug': road_classes,
    'deepglobe': road_classes,
    'massachusetts': road_classes,
    'spacenet': road_classes,
    'wbs_si': water_classes,
}


def _validate_protocol(args):
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    if world_size != 1:
        raise RuntimeError(
            'Strict CTTA requires WORLD_SIZE=1. DDP partitions the temporal stream; '
            'parallelize independent datasets/runs across GPUs instead.'
        )
    if args.batch_size != 1:
        raise ValueError(f'Strict CTTA requires --batch_size 1, got {args.batch_size}.')
    if args.reset_mode != 'source' and args.tta_steps <= 0:
        raise ValueError('Non-source CTTA modes require --tta_steps > 0.')


def _build_transform():
    normalize = transforms.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711],
    )
    return transforms.Compose([transforms.ToTensor(), normalize])


def _build_model_and_optimizer(args, classnames, device):
    model = get_shift_model(args, classnames).to(device)
    model.eval()

    trainable_param = []
    if args.text_shift:
        trainable_param.extend(model.text_shifter.parameters())
        if hasattr(model, 'text_shifter_visual'):
            trainable_param.extend(model.text_shifter_visual.parameters())

    optimizer = None
    if args.tta_steps > 0:
        other_params = [p for p in trainable_param if p is not model.alpha]
        param_groups = []
        if other_params:
            param_groups.append({'params': other_params, 'lr': args.lr})
        param_groups.append({'params': [model.alpha], 'lr': args.lr})
        optimizer = torch.optim.AdamW(param_groups)

    return model, optimizer


def _evaluate_domain(args, set_id, corruption, model, optimizer, scaler, state, device):
    dataset = CorruptedRemoteDataset(
        set_id=set_id,
        transform=_build_transform(),
        data_root=args.data,
        mode=args.dataset_mode,
        num_classes=args.num_classes,
        args=args,
        corruption=corruption,
        corruption_severity=args.corruption_severity,
        daf_root=args.daf_root,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
        drop_last=False,
    )

    domain_results = []
    for image_name, images, ori_shape, target in tqdm(
        loader, total=len(loader), desc=f'{set_id}:{corruption}'
    ):
        state.before_sample()
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        if state.should_adapt:
            test_time_tuning(
                image_name[0], model, images, ori_shape, optimizer, scaler, args
            )

        with torch.no_grad(), torch.cuda.amp.autocast(enabled=device.type == 'cuda'):
            if args.loss_prompt == 'True':
                mask_pred, _, _ = model(images, ori_shape, image_name[0], args, test=True)
            else:
                mask_pred, _ = model(images, ori_shape, image_name[0], args, test=True)

        target = target.to(mask_pred.device)
        for mask_i, output_i in zip(target, mask_pred):
            domain_results.append(
                intersect_and_union(
                    output_i.int().contiguous().clone(),
                    mask_i.int().contiguous(),
                    args.num_classes,
                    ignore_index=255,
                )
            )
    return domain_results


def run_dataset(args, set_id, device):
    if set_id not in CLASSES_DICT:
        raise KeyError(f'Unknown TMPA remote dataset: {set_id}')

    args.test_sets = set_id
    args.dataset_name = set_id
    classnames = CLASSES_DICT[set_id]
    corruptions = expand_corruptions(args.corruptions_list)

    model, optimizer = _build_model_and_optimizer(args, classnames, device)
    scaler = torch.cuda.amp.GradScaler(init_scale=1e3, enabled=device.type == 'cuda')
    state = AdaptationStateController(model, optimizer, args, args.reset_mode)
    state.before_stream()

    stream_results = []
    domains = []
    for domain_index, corruption in enumerate(corruptions):
        state.before_domain(domain_index)
        domain_results = _evaluate_domain(
            args, set_id, corruption, model, optimizer, scaler, state, device
        )
        stream_results.extend(domain_results)
        domain_metrics = summarize_results(domain_results)
        domains.append({
            'index': domain_index,
            'corruption': corruption,
            'num_samples': len(domain_results),
            'metrics': domain_metrics,
        })
        print(
            f"[{set_id}] {corruption}: "
            f"mIoU={domain_metrics.get('mIoU', float('nan')):.2f}, "
            f"mAcc={domain_metrics.get('mAcc', float('nan')):.2f}"
        )

    return {
        'dataset': set_id,
        'protocol': {
            'reset_mode': args.reset_mode,
            'corruptions': corruptions,
            'corruption_severity': args.corruption_severity,
            'tta_steps': args.tta_steps,
            'batch_size': 1,
            'stream_semantics': 'strict sequential single-process',
        },
        'domains': domains,
        'stream_metrics': summarize_results(stream_results),
        'stream_progress': summarize_progress(stream_results, args.ctta_progress),
    }


def main(args):
    _validate_protocol(args)
    set_random_seed(args.seed)

    if not torch.cuda.is_available():
        raise RuntimeError('TMPA CTTA evaluation currently requires CUDA.')
    device = torch.device(f'cuda:{args.gpu}')
    torch.cuda.set_device(device)

    output_dir = Path(args.ctta_result_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = args.test_sets.split('/')
    all_results = {}
    for set_id in datasets:
        result = run_dataset(args, set_id, device)
        all_results[set_id] = result
        out_path = output_dir / f'{set_id}_{args.reset_mode}_sev{args.corruption_severity}.json'
        with out_path.open('w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f'[INFO] CTTA result saved to {out_path}')

    summary_path = output_dir / f'all_{args.reset_mode}_sev{args.corruption_severity}.json'
    with summary_path.open('w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f'[INFO] Combined CTTA results saved to {summary_path}')


if __name__ == '__main__':
    main(parse_args())
