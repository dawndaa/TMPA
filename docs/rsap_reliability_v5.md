# Three reliability ablations on v5

v5 is based on v4 commit `e339e9aef9b31691a2d5914cd3485884bf525982`.
Add one of the following switches to your existing Full RSAP+SDR command:

- **`--no_rsap_reliability`**: disable reliability only in RSAP.
- **`--no_sdr_reliability`**: disable reliability only in SDR.
- **`--no_reliability`**: disable reliability in both; alias for the two switches above.

The RSAP-only switch requires `--module_rsap_v1`; the SDR-only switch requires
`--loss_sdr`; the joint switch requires both. Without these switches, the
original v4 Full numerical path and result filenames remain.

## Four configurations

Use the same Full command with `--module_rsap_v1 --text_adjust True --loss_sdr`.
Keep the same `sdr_min_weight` and all other settings as your Full run.

| Configuration | Extra settings |
|---|---|
| Full | No additional switch |
| Full w/o RSAP reliability | `--no_rsap_reliability` |
| Full w/o SDR reliability | `--no_sdr_reliability` |
| Full w/o both reliabilities | `--no_reliability` |

Combining `--no_rsap_reliability --no_sdr_reliability` gives the same behavior
and result tag as `--no_reliability`. Setting `--sdr_min_weight 1.0` is also
numerically equivalent to uniform GSC, but the new SDR switch makes the
ablation explicit. `--rsap_gamma 0` removes only disagreement penalization and
is not equivalent to disabling RSAP reliability.

## What changes in RSAP

| Component | Default RSAP | `--no_rsap_reliability` |
|---|---|---|
| Cat-Prompt prediction | Multi-description bank | Same |
| Token class assignment | Frozen consensus argmax | Same |
| K-token selection per class | Highest reliability | Fixed evenly spaced candidate positions |
| Visual prototype | Reliability-weighted mean | Equal-weight mean |
| Fusion | `clamp(alpha * r_class, 0, 1)` | `clamp(alpha, 0, 1)` for present classes |
| Absent classes | Zero visual gate | Same |
| SDR reliability map | Original `consensus * exp(-gamma * disagreement)` | Same |

Candidates are ordered by flattened visual-token position. For `n` candidates,
use `k=min(rsap_topk,n)` positions `floor((2*j+1)*n/(2*k))`, `j=0,...,k-1`.
This is deterministic stratified selection, retains the K-token budget and
does not consume random state. If fewer than K candidates exist, use all of
them. No confidence or disagreement score ranks or weights the prototypes.

Both the online model and its frozen source-state copy use the selected RSAP
mode, following the existing v4 reference definition. With only RSAP reliability
disabled, SDR still uses the original detached reliability map, including its
spatial variation and dependence on `rsap_gamma`. Predictions and SDR loss
values can change because RSAP prediction changes; the SDR weighting rule
itself is preserved.

`--no_sdr_reliability` calls uniform symmetric-KL source consistency directly.
It preserves `lamb_sdr` and the reference model, while ignoring the configured
`sdr_min_weight` (including zero). It does not change RSAP selection or fusion.

With both reliabilities disabled, the map is still computed for diagnostics,
but neither RSAP nor SDR uses its scores. It is the real estimate, not an
all-one placeholder. `rsap_gamma` affects this diagnostic map only, not the
ablated prediction or loss; `sdr_min_weight` also no longer affects the loss.

## Run

Add the flag to your existing Full command, with all other experiment settings
kept identical. For example, OpenEarthMap Val with the v4/v5 continual protocol:

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
  --no_rsap_reliability \
  --reset_mode continual --corruptions_list common --corruption_severity 5 \
  --ctta_result_dir save_result/rsap_sdr_v5/ctta_wo_rsap_reliability
```

For the other two ablations, replace `--no_rsap_reliability` with
`--no_sdr_reliability` or `--no_reliability`, and choose a corresponding output
directory. For UDD5 or another dataset, add the selected switch to its existing
Full command. Do not change dataset split, corruption order, severity, seed,
learning rate, model weights or resolution between configurations.

Result filenames gain `wo-rsap-reliability`, `wo-sdr-reliability`, or
`full-wo-reliability`. JSON metadata records both disabled flags, RSAP's
`reliability_enabled`, and SDR's `pixel_weighting` / `effective_min_pixel_weight`.
The configured weight floor is retained for provenance even when inactive.

## Checks

```bash
python -m unittest discover -s tests -p 'test_rsap_reliability.py' -v
```

Tests use CPU tensors, require PyTorch, and need no datasets or model weights.
They check that the RSAP ablation preserves SDR's real reliability map while
removing reliability from RSAP mining/gating; that the SDR ablation leaves RSAP
forward results unchanged and uses uniform GSC; and that all three switches,
their alias combinations, gradients, absent classes and result tags work.
No dataset evaluation or mIoU/mAcc results are included in this change.
