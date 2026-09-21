"""Strict single-stream CTTA evaluation for TMPA remote-sensing datasets.

Keeps TMPA's remote datasets/model/adaptation, reuses DAF corruptions, and
changes only the temporal reset policy. Phase-2 stabilizers are opt-in so the
Phase-1 TMPA-Continual baseline remains unchanged by default.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
import torchvision.transforms as transforms
from tqdm import tqdm

from args import parse_args
from ctta.adaptation import (
    build_source_model,
    needs_ctta_loop,
    needs_source_model,
    summarize_loss_reports,
    test_time_tuning_ctta,
)
from ctta.corruptions import expand_corruptions
from ctta.dataset import CorruptedRemoteDataset
from ctta.metrics import summarize_progress, summarize_results
from ctta.modules.safs import TemporalSAFSGate
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
    if args.reset_mode == 'source' and needs_ctta_loop(args):
        raise ValueError('DAF-derived stabilization modules cannot be used with --reset_mode source.')
    if args.loss_prompt_feat_cons and not args.text_shift:
        raise ValueError('--loss_prompt_feat_cons requires --text_shift; otherwise the prompt feature is fixed.')
    if args.module_safs:
        if args.safs_window < 2:
            raise ValueError('--safs_window must be >= 2.')
        if args.safs_warmup < 1 or args.safs_warmup > args.safs_window:
            raise ValueError('--safs_warmup must be in [1, safs_window].')
        if args.alpha_safs < 0:
            raise ValueError('--alpha_safs must be non-negative.')


def _build_transform():
    normalize = transforms.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711],
    )
    return transforms.Compose([transforms.ToTensor(), normalize])


def _build_model_and_optimizer(args, classnames, device):
    model = get_shift_model(args, classnames).to(device)
    model.eval()

    # RSAP/SDR adapts only prompt-side parameters and alpha. Freeze the full
    # model before the first forward so the visual backbone/upsampler never
    # build a backward graph on high-resolution samples.
    model.requires_grad_(False)

    trainable_param = []
    if args.text_shift:
        # text_shifter is the actual text-offset parameterization used by the
        # current forward path. The legacy text_shifter_visual object is kept
        # for checkpoint/reset compatibility but is not optimized in v4.
        model.text_shifter.requires_grad_(True)
        trainable_param.extend(model.text_shifter.parameters())

    alpha_is_active = bool(
        args.module_rsap_v1
        or (args.module_visual_guidance and args.text_adjust == 'True')
    )
    model.alpha.requires_grad_(alpha_is_active)

    optimizer = None
    if args.tta_steps > 0:
        param_groups = []
        if trainable_param:
            param_groups.append({'params': trainable_param, 'lr': args.lr})
        if alpha_is_active:
            param_groups.append({'params': [model.alpha], 'lr': args.lr})
        if param_groups:
            optimizer = torch.optim.AdamW(param_groups)

    source_model = build_source_model(model, args)
    return model, optimizer, source_model


def _build_safs_gate(args):
    if not args.module_safs:
        return None
    return TemporalSAFSGate(
        alpha=args.alpha_safs,
        window=args.safs_window,
        warmup=args.safs_warmup,
    )


def _evaluate_domain(
    args,
    set_id,
    corruption,
    model,
    optimizer,
    scaler,
    state,
    device,
    source_model=None,
    safs_gate=None,
):
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
    loss_reports = []
    for image_name, images, ori_shape, target in tqdm(
        loader, total=len(loader), desc=f'{set_id}:{corruption}'
    ):
        state.before_sample()
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        if state.should_adapt:
            if needs_ctta_loop(args):
                loss_reports.extend(
                    test_time_tuning_ctta(
                        image_name[0],
                        model,
                        images,
                        ori_shape,
                        optimizer,
                        scaler,
                        args,
                        source_model=source_model,
                        safs_gate=safs_gate,
                    )
                )
            else:
                # Keep the original TMPA adaptation path bit-for-bit when no
                # Phase-2 stabilizer is enabled.
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
    return domain_results, summarize_loss_reports(loss_reports)


def _tmpa_module_config(args):
    return {
        'cat_prompt': {
            'enabled': bool(args.module_cat_prompt),
            'mode': 'multi_description' if args.module_cat_prompt else 'single_category_name',
            'source_prompt_file': getattr(args, 'cat_prompt_source_path', args.name_path),
            'definition': (
                'Controls only the segmentation-prediction prompt bank in v4. '
                'SDR reliability may still use the frozen auxiliary multi-description bank.'
            ),
        },
        'reliability_prompt_bank': {
            'enabled': bool(args.module_rsap_v1 or args.loss_sdr),
            'source_prompt_file': getattr(
                args, 'reliability_prompt_path',
                getattr(args, 'cat_prompt_source_path', args.name_path)
            ),
            'prediction_uses_same_bank': bool(args.module_cat_prompt),
            'definition': (
                'Frozen multi-description text bank used only to estimate detached semantic '
                'reliability. When RSAP is disabled, this bank does not participate in '
                'segmentation prediction.'
            ),
        },
        'visual_guidance': {
            'enabled': bool(args.module_visual_guidance and args.text_adjust == 'True'),
            'module_switch': bool(args.module_visual_guidance),
            'legacy_text_adjust': args.text_adjust,
            'definition': (
                'TMPA image-guided prompt adjustment only. This switch does not control '
                'text_shift, optimizer creation, tta_steps, or reset semantics.'
            ),
        },
        'rsap_v1': {
            'enabled': bool(args.module_rsap_v1),
            'topk': args.rsap_topk,
            'gamma': args.rsap_gamma,
            'definition': (
                'Reliability-aware Scene-Adaptive Prompting: frozen Cat-Prompt consensus '
                'estimates target reliability, mines class visual prototypes, and gates '
                'text/visual calibration independently of legacy Visual Guidance.'
            ),
        },
        'tta_update': {
            'text_shift': bool(args.text_shift),
            'do_shift': bool(args.do_shift),
            'do_scale': bool(args.do_scale),
            'do_film': bool(args.do_film),
            'tta_steps': args.tta_steps,
            'reset_mode': args.reset_mode,
            'definition': (
                'Existing TMPA test-time optimization path; controlled independently from '
                'Visual Guidance by reset_mode, tta_steps and the legacy shift/scale flags.'
            ),
        },
    }


def _stabilization_config(args):
    return {
        'source_consistency': {
            'enabled': bool(args.loss_src_cons),
            'weight': args.lamb_src_cons,
            'definition': 'DAF-style symmetric KL against frozen source prediction',
        },
        'sdr': {
            'enabled': bool(args.loss_sdr),
            'weight': args.lamb_sdr,
            'min_pixel_weight': args.sdr_min_weight,
            'definition': (
                'Reliability-guided GSC: preserve symmetric-KL source consistency while '
                'redistributing pixel weights as w_min + (1-w_min)*(1-r). Reliability is '
                'read from the frozen multi-description bank and is independent of whether '
                'that bank is used by the segmentation prediction branch.'
            ),
        },
        'prompt_feature_consistency': {
            'enabled': bool(args.loss_prompt_feat_cons),
            'weight': args.lamb_prompt_feat_cons,
            'type': args.prompt_feat_cons_type,
            'definition': (
                'TMPA prompt/text-space counterpart to DAF visual feature consistency; '
                'DAF visual feature consistency is not copied directly because TMPA freezes visual features'
            ),
        },
        'cmac': {
            'enabled': bool(args.loss_cmac),
            'weight': args.lamb_cmac,
            'definition': (
                'DAF-style directional source anchoring mapped to TMPA pixel class probabilities: '
                'penalize moving away from the frozen-source assigned class or toward non-anchor classes'
            ),
        },
        'diversity': {
            'enabled': bool(args.loss_div),
            'weight': args.lamb_div,
            'definition': (
                'DAF-style negative entropy of the marginal class distribution across pixels; '
                'minimization discourages collapse to a small set of predicted classes'
            ),
        },
        'safs': {
            'enabled': bool(args.module_safs),
            'alpha': args.alpha_safs,
            'window': args.safs_window,
            'warmup': args.safs_warmup,
            'definition': (
                'Streaming analogue of DAF SAFS for strict batch-size-one CTTA: '
                'gate optimizer updates using prediction drift against a frozen source anchor '
                'and a rolling mean - alpha * std threshold'
            ),
        },
    }


def _result_tag(args):
    parts = [args.reset_mode, f'sev{args.corruption_severity}']
    # Keep legacy filenames unchanged for the default full-TMPA configuration.
    if not args.module_cat_prompt:
        parts.append('nocatprompt')
    if not args.module_visual_guidance:
        parts.append('novisguide')
    if args.module_rsap_v1:
        gamma = f'{args.rsap_gamma:g}'.replace('.', 'p')
        parts.append(f'rsapv1-k{args.rsap_topk}-g{gamma}')
    if args.loss_src_cons:
        parts.append(f'srccons{args.lamb_src_cons:g}'.replace('.', 'p'))
    if args.loss_sdr:
        weight = f'{args.lamb_sdr:g}'.replace('.', 'p')
        floor = f'{args.sdr_min_weight:g}'.replace('.', 'p')
        parts.append(f'sdr{weight}-wmin{floor}')
    if args.loss_prompt_feat_cons:
        weight = f'{args.lamb_prompt_feat_cons:g}'.replace('.', 'p')
        parts.append(f'pfeat-{args.prompt_feat_cons_type}-{weight}')
    if args.loss_cmac:
        parts.append(f'cmac{args.lamb_cmac:g}'.replace('.', 'p'))
    if args.loss_div:
        parts.append(f'div{args.lamb_div:g}'.replace('.', 'p'))
    if args.module_safs:
        alpha = f'{args.alpha_safs:g}'.replace('.', 'p')
        parts.append(f'safs-a{alpha}-w{args.safs_window}-u{args.safs_warmup}')
    return '_'.join(parts)


def run_dataset(args, set_id, device):
    if set_id not in CLASSES_DICT:
        raise KeyError(f'Unknown TMPA remote dataset: {set_id}')

    args.test_sets = set_id
    args.dataset_name = set_id
    classnames = CLASSES_DICT[set_id]
    corruptions = expand_corruptions(args.corruptions_list)

    model, optimizer, source_model = _build_model_and_optimizer(args, classnames, device)
    scaler = torch.cuda.amp.GradScaler(init_scale=1e3, enabled=device.type == 'cuda')
    state = AdaptationStateController(model, optimizer, args, args.reset_mode)
    state.before_stream()
    safs_gate = _build_safs_gate(args)

    stream_results = []
    domains = []
    for domain_index, corruption in enumerate(corruptions):
        state.before_domain(domain_index)
        if safs_gate is not None and args.reset_mode in ('episodic', 'domain'):
            safs_gate.reset()

        domain_results, loss_summary = _evaluate_domain(
            args,
            set_id,
            corruption,
            model,
            optimizer,
            scaler,
            state,
            device,
            source_model=source_model,
            safs_gate=safs_gate,
        )
        stream_results.extend(domain_results)
        domain_metrics = summarize_results(domain_results)
        domains.append({
            'index': domain_index,
            'corruption': corruption,
            'num_samples': len(domain_results),
            'metrics': domain_metrics,
            'adaptation_losses': loss_summary,
        })
        print(
            f"[{set_id}] {corruption}: "
            f"mIoU={domain_metrics.get('mIoU', float('nan')):.2f}, "
            f"mAcc={domain_metrics.get('mAcc', float('nan')):.2f}"
        )
        if loss_summary:
            print(f'[{set_id}] {corruption} adaptation losses: {loss_summary}')

    return {
        'dataset': set_id,
        'protocol': {
            'reset_mode': args.reset_mode,
            'corruptions': corruptions,
            'corruption_severity': args.corruption_severity,
            'tta_steps': args.tta_steps,
            'batch_size': 1,
            'stream_semantics': 'strict sequential single-process',
            'adaptation_enabled': bool(args.reset_mode != 'source' and optimizer is not None),
            'tmpa_modules': _tmpa_module_config(args),
            'stabilization': _stabilization_config(args),
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
    result_tag = _result_tag(args)
    for set_id in datasets:
        result = run_dataset(args, set_id, device)
        all_results[set_id] = result
        out_path = output_dir / f'{set_id}_{result_tag}.json'
        with out_path.open('w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f'[INFO] CTTA result saved to {out_path}')
        torch.cuda.empty_cache()

    summary_path = output_dir / f'all_{result_tag}.json'
    with summary_path.open('w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f'[INFO] Combined CTTA results saved to {summary_path}')


if __name__ == '__main__':
    main(parse_args())
