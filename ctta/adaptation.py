"""TMPA adaptation loop with optional DAF-derived CTTA stabilizers."""

from __future__ import annotations

from copy import deepcopy

import torch

from ctta.modules.consistency import (
    prompt_feature_consistency,
    source_prediction_consistency,
)
from run_utils import avg_entropy_seg, loss_prompt_entropy


def needs_source_model(args) -> bool:
    return bool(args.loss_src_cons or args.loss_prompt_feat_cons)


def build_source_model(model, args):
    """Create the frozen source anchor used by consistency losses."""
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
):
    """Run TMPA test-time tuning with optional source/prompt consistency."""
    if optimizer is None:
        return []
    if needs_source_model(args) and source_model is None:
        raise ValueError('Consistency loss requested but no frozen source model was provided.')

    reports = []
    autocast_enabled = inputs.is_cuda

    for _ in range(args.tta_steps):
        with torch.cuda.amp.autocast(enabled=autocast_enabled):
            source_prob_maps = None
            if args.loss_src_cons:
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

            prompt_cons_loss = base_loss.new_zeros(())
            if args.loss_prompt_feat_cons:
                prompt_cons_loss = prompt_feature_consistency(
                    model,
                    source_model,
                    args,
                    loss_type=args.prompt_feat_cons_type,
                )

            total_loss = (
                base_loss
                + args.lamb_src_cons * src_cons_loss
                + args.lamb_prompt_feat_cons * prompt_cons_loss
            )

        if total_loss.requires_grad and any(p.requires_grad for p in model.parameters()):
            optimizer.zero_grad()
            scaler.scale(total_loss).backward(retain_graph=True)
            scaler.step(optimizer)
            scaler.update()

        reports.append({
            'total': float(total_loss.detach().item()),
            'base': float(base_loss.detach().item()),
            'entropy': float(loss_entropy.detach().item()),
            'prompt': None if loss_prompt is None else float(loss_prompt.detach().item()),
            'source_consistency': float(src_cons_loss.detach().item()),
            'prompt_feature_consistency': float(prompt_cons_loss.detach().item()),
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
    return summary
