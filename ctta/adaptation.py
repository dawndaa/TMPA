"""TMPA adaptation loop with optional DAF-derived CTTA stabilizers."""

from __future__ import annotations

from copy import deepcopy

import torch

from ctta.modules.cmac import cmac_source_anchor_loss
from ctta.modules.consistency import (
    prompt_feature_consistency,
    reliability_guided_source_consistency,
    source_prediction_consistency,
)
from ctta.modules.diversity import diversity_loss
from ctta.modules.safs import prediction_shift_score
from run_utils import avg_entropy_seg, loss_prompt_entropy


def needs_source_model(args) -> bool:
    return bool(
        args.loss_src_cons
        or args.loss_sdr
        or args.loss_prompt_feat_cons
        or args.loss_cmac
        or args.module_safs
    )


def needs_ctta_loop(args) -> bool:
    """Whether adaptation must use the extended CTTA loss loop."""
    return bool(needs_source_model(args) or args.loss_div)


def build_source_model(model, args):
    """Create the frozen source anchor used by source-anchored stabilizers."""
    if not needs_source_model(args):
        return None

    source_model = deepcopy(model)
    with torch.no_grad():
        source_model.reset(args)
    source_model.requires_grad_(False)
    source_model.eval()
    return source_model


def _forward_source(source_model, image_name, inputs, ori_shape, args):
    with torch.no_grad():
        if args.loss_prompt == 'True':
            _, source_prob_maps, _ = source_model(
                inputs, ori_shape, image_name, args, test=False
            )
        else:
            _, source_prob_maps = source_model(
                inputs, ori_shape, image_name, args, test=False
            )
    return source_prob_maps.detach()


def _forward_adapted(model, image_name, inputs, ori_shape, args):
    if args.loss_prompt == 'True':
        mask_pred, seg_prob_maps, all_seg_logits = model(
            inputs, ori_shape, image_name, args
        )
        loss_entropy = avg_entropy_seg(seg_prob_maps, args.tps_entropy_scale)
        loss_prompt = loss_prompt_entropy(
            all_seg_logits,
            seg_prob_maps,
            logit_scale=args.prompt_logit_scale,
            args=args,
        )
        base_loss = (loss_prompt + loss_entropy) / 2
    else:
        mask_pred, seg_prob_maps = model(inputs, ori_shape, image_name, args)
        loss_entropy = avg_entropy_seg(seg_prob_maps, args.tps_entropy_scale)
        loss_prompt = None
        base_loss = loss_entropy

    return mask_pred, seg_prob_maps, base_loss, loss_entropy, loss_prompt


def test_time_tuning_ctta(
    image_name,
    model,
    inputs,
    ori_shape,
    optimizer,
    scaler,
    args,
    source_model=None,
    safs_gate=None,
):
    """Run TMPA test-time tuning with optional DAF-derived stabilizers."""
    if optimizer is None:
        return []
    if needs_source_model(args) and source_model is None:
        raise ValueError('A source-anchored CTTA module was requested but no frozen source model was provided.')
    if args.module_safs and safs_gate is None:
        raise ValueError('SAFS was enabled but no temporal SAFS gate was provided.')

    reports = []
    autocast_enabled = inputs.is_cuda

    for _ in range(args.tta_steps):
        with torch.cuda.amp.autocast(enabled=autocast_enabled):
            source_prob_maps = None
            if args.loss_src_cons or args.loss_sdr or args.loss_cmac or args.module_safs:
                source_prob_maps = _forward_source(
                    source_model, image_name, inputs, ori_shape, args
                )

            _, seg_prob_maps, base_loss, loss_entropy, loss_prompt = _forward_adapted(
                model, image_name, inputs, ori_shape, args
            )

            src_cons_loss = base_loss.new_zeros(())
            if args.loss_src_cons:
                src_cons_loss = source_prediction_consistency(
                    seg_prob_maps, source_prob_maps
                )

            sdr_loss = base_loss.new_zeros(())
            if args.loss_sdr:
                reliability_map = getattr(model, 'last_rsap_reliability_map', None)
                if reliability_map is None:
                    raise RuntimeError(
                        'SDR requires the RSAP reliability map from the adapted forward pass.'
                    )
                sdr_loss = reliability_guided_source_consistency(
                    seg_prob_maps,
                    source_prob_maps,
                    reliability_map,
                    min_weight=args.sdr_min_weight,
                )

            prompt_cons_loss = base_loss.new_zeros(())
            if args.loss_prompt_feat_cons:
                prompt_cons_loss = prompt_feature_consistency(
                    model,
                    source_model,
                    args,
                    loss_type=args.prompt_feat_cons_type,
                )

            cmac_loss = base_loss.new_zeros(())
            if args.loss_cmac:
                cmac_loss = cmac_source_anchor_loss(
                    seg_prob_maps,
                    source_prob_maps,
                )

            div_loss = base_loss.new_zeros(())
            if args.loss_div:
                div_loss = diversity_loss(seg_prob_maps)

            total_loss = (
                base_loss
                + args.lamb_src_cons * src_cons_loss
                + args.lamb_sdr * sdr_loss
                + args.lamb_prompt_feat_cons * prompt_cons_loss
                + args.lamb_cmac * cmac_loss
                + args.lamb_div * div_loss
            )

        safs_info = {
            'keep': 1.0,
            'score': None,
            'threshold': None,
            'history_mean': None,
            'history_std': None,
            'history_size': None,
        }
        if args.module_safs:
            shift = prediction_shift_score(seg_prob_maps, source_prob_maps)
            safs_info = safs_gate.decide(shift)

        did_update = bool(safs_info['keep'])
        if did_update and total_loss.requires_grad and any(p.requires_grad for p in model.parameters()):
            optimizer.zero_grad()
            scaler.scale(total_loss).backward(retain_graph=True)
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.zero_grad()

        reports.append({
            'total': float(total_loss.detach().item()),
            'base': float(base_loss.detach().item()),
            'entropy': float(loss_entropy.detach().item()),
            'prompt': None if loss_prompt is None else float(loss_prompt.detach().item()),
            'source_consistency': float(src_cons_loss.detach().item()),
            'sdr': float(sdr_loss.detach().item()),
            'prompt_feature_consistency': float(prompt_cons_loss.detach().item()),
            'cmac': float(cmac_loss.detach().item()),
            'diversity': float(div_loss.detach().item()),
            'safs_keep': float(did_update),
            'safs_shift': safs_info['score'],
            'safs_threshold': safs_info['threshold'],
            'safs_history_mean': safs_info['history_mean'],
            'safs_history_std': safs_info['history_std'],
            'safs_history_size': safs_info['history_size'],
        })

    return reports


def summarize_loss_reports(reports):
    if not reports:
        return {}

    summary = {}
    for key in reports[0]:
        values = [report[key] for report in reports if report[key] is not None]
        if values:
            summary[key] = sum(values) / len(values)

    if 'safs_keep' in summary:
        summary['safs_filter_rate'] = 1.0 - summary['safs_keep']
    return summary
