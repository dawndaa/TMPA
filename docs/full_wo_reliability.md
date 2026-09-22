# Full w/o reliability (based on RSAP-SDR v4)

Base: `feature/rsap-sdr-v4`, commit `e339e9aef9b31691a2d5914cd3485884bf525982`.

Add **`--no_reliability`** to the same Full RSAP+SDR command used for your
existing experiment. The flag requires `--module_rsap_v1 --loss_sdr` and is off
by default. Existing Full runs retain their original numerical path.

## Exact ablation definition

| Component | Full | Full w/o reliability |
|---|---|---|
| Cat-Prompt prediction | Multi-description | Same |
| Token class assignment | Frozen multi-description consensus argmax | Same |
| K token selection per class | Highest reliability | Fixed evenly spaced candidate positions |
| Visual prototype | Reliability-weighted mean | Equal-weight mean |
| Fusion coefficient | `clamp(alpha * class_reliability, 0, 1)` | `clamp(alpha, 0, 1)` for present classes |
| Absent classes | Zero prototype and zero visual gate | Same |
| Source consistency | Reliability-weighted symmetric KL | Uniform symmetric KL (GSC) |
| Source anchor / loss multiplier | Frozen source-state copy / `lamb_sdr` | Same |
| Trainable parameters and CTTA protocol | v4 configuration | Same |

The ablation removes **all uses of reliability**, including Top-K ranking.
It preserves consensus only to assign each token to a semantic class.
For `n` candidates in raster order and `k=min(rsap_topk,n)`, choose candidate
positions `floor((2*j+1)*n/(2*k))`, `j=0,...,k-1`. This preserves the token
budget, uses no confidence scores, introduces no random state, and gives the
same selection to online/reference branches. It is deterministic stratified
selection, not random sampling. If `n<rsap_topk`, use all candidates.

In this mode the exported reliability map is an all-one placeholder; do not
interpret or plot it as estimated reliability. No prompt disagreement is
computed. `rsap_gamma` and `sdr_min_weight` no longer affect computation.
Uniform GSC is called directly, including when `sdr_min_weight=0`.

Setting only `rsap_gamma=0` removes disagreement but retains confidence-based
reliability. Setting only `sdr_min_weight=1` removes SDR weighting but leaves
RSAP reliability active. Neither is this full ablation.

Both online and frozen source-state models receive the flag. The reference
therefore uses the same ablated architecture at source state, following v4's
reference definition. Cat-Prompt aggregation, entropy objective, parameter
initialization, learning rate and update/reset rules are unchanged.

## Run on OpenEarthMap Val

Use your actual dataset root and preserve all settings from your existing Full
run. The example below follows the v4 continual protocol:

```bash
CUDA_VISIBLE_DEVICES=0 python ctta_eval_remote.py /path/to/data \
  --test_sets openearthmap --dataset_mode val \
  -a ViT-B/16 -b 1 --gpu 0 --resolution 448 \
  --lr 1e-4 --tta_steps 1 --num_classes 9 \
  --name_path ./configs/cls_openearthmap.txt \
  --module_cat_prompt --no_module_visual_guidance \
  --module_rsap_v1 --rsap_topk 3 --rsap_gamma 1.0 \
  --text_adjust True --text_shift --do_shift --per_label \
  --loss_sdr --lamb_sdr 1.0 --sdr_min_weight 0.5 \
  --no_reliability \
  --reset_mode continual --corruptions_list common --corruption_severity 5 \
  --ctta_result_dir save_result/rsap_sdr_v4/ctta_full_wo_reliability
```

For Full, remove `--no_reliability` and select a separate result directory.
For UDD5, use your existing UDD5 Full command and add the same flag.
Keep dataset split, corruption order, severity, seed, resolution, model weights
and all other hyperparameters identical within each comparison.

Result filenames include `full-wo-reliability`. JSON metadata records
`reliability_ablation.enabled=true`, `role=consensus_only`, and SDR
`pixel_weighting=uniform` / `effective_min_pixel_weight=1.0`. The configured
gamma and weight floor remain recorded for provenance but are inactive.

## Regression checks

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
```

The new tests check flag validation, distinct result names, effective metadata,
equal-weight prototypes, absent classes, fixed sampling, gamma independence,
unchanged random state, and the default Full path. Numerical tests require
PyTorch; no pretrained weights or GPU are needed for the mining tests.

This change does not include dataset evaluation or new mIoU/mAcc results.
