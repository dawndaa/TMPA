# RSAP-SDR v4

This document is the source of truth for branch `feature/rsap-sdr-v4`.

For backward compatibility, the existing CLI switch `--module_rsap_v1` is retained; on this branch it enables the current RSAP implementation described below.

## 1. What changed in v4

v4 decouples the **segmentation-prediction prompt bank** from the
**multi-description reliability bank**.

The purpose is to make RSAP and SDR independently switchable without changing
the Full RSAP+SDR behavior.

### Prediction prompt bank

Controlled by `--module_cat_prompt` / `--no_module_cat_prompt`.

- `--module_cat_prompt`: segmentation prediction uses the original
  multi-description Cat-Prompt bank.
- `--no_module_cat_prompt`: segmentation prediction uses exactly one category
  prompt per class. A temporary single-prompt file is created automatically.

### Reliability prompt bank

The original multi-description file passed by `--name_path` is preserved as
a frozen auxiliary reliability bank.

It is used only when RSAP or SDR needs semantic reliability:

```text
frozen multi-description bank
        +
frozen visual tokens
        |
        v
multi-description consensus / disagreement
        |
        v
reliability map r
```

When RSAP is disabled but SDR is enabled, this bank **does not participate in
segmentation prediction**. It only produces the detached reliability map.

This gives a strict SDR-only ablation.

## 2. Module boundaries

### RSAP

RSAP contains the prediction-side multi-description semantic path:

```text
Cat-Prompt prediction
  -> multi-description reliability
  -> reliable Top-K visual prototypes
  -> reliability-gated text/visual calibration
  -> dense segmentation prediction
```

RSAP requires:

```text
--module_cat_prompt
--module_rsap_v1
--text_adjust True
--no_module_visual_guidance
```

### SDR

SDR consumes:

```text
online prediction p
frozen source-state prediction q
reliability map r
```

and optimizes reliability-guided symmetric-KL source consistency:

```text
w = w_min + (1 - w_min) * (1 - r)
L_SDR = sum(w * D_symKL(p, q)) / sum(w)
```

SDR no longer requires Cat-Prompt to be used by the prediction branch.

With RSAP off, SDR automatically obtains `r` from the frozen auxiliary
multi-description bank.

## 3. Four ablation states

| RSAP | SDR | Prediction prompts | Reliability source |
|---|---|---|---|
| off | off | single prompt | disabled |
| on | off | Cat-Prompt | RSAP multi-description bank |
| off | on | single prompt | frozen auxiliary multi-description bank |
| on | on | Cat-Prompt | shared RSAP multi-description bank |

The Full model is the final row.

## 4. Important prompt-routing rule

Always pass the original **multi-description** dataset prompt file to
`--name_path`, including SDR-only runs.

Example:

```bash
--name_path ./configs/cls_loveda.txt
```

For `--no_module_cat_prompt`, v4 automatically:

1. preserves that original file as the reliability bank;
2. creates a temporary one-prompt-per-class file for segmentation prediction.

Do not manually replace `--name_path` with a single-prompt file for SDR-only,
otherwise the auxiliary reliability bank will no longer contain multiple
descriptions.

## 5. LoveDA clean ablations

Common settings:

```bash
export DATA_ROOT=/path/to/data
export GPU=0
```

### 5.1 w/o RSAP, SDR

Single-prompt prediction, text adaptation only.

```bash
CUDA_VISIBLE_DEVICES=$GPU python ctta_eval_remote.py "$DATA_ROOT" \
  --test_sets loveda -a ViT-B/16 -b 1 --gpu 0 --resolution 448 \
  --lr 1e-4 --tta_steps 1 --num_classes 7 \
  --name_path ./configs/cls_loveda.txt \
  --no_module_cat_prompt --no_module_visual_guidance \
  --text_shift --do_shift --per_label \
  --reset_mode domain \
  --corruptions_list original --corruption_severity 5 \
  --ctta_result_dir save_result/rsap_sdr_v4/clean_none
```

### 5.2 RSAP only (w/o SDR)

```bash
CUDA_VISIBLE_DEVICES=$GPU python ctta_eval_remote.py "$DATA_ROOT" \
  --test_sets loveda -a ViT-B/16 -b 1 --gpu 0 --resolution 448 \
  --lr 1e-4 --tta_steps 1 --num_classes 7 \
  --name_path ./configs/cls_loveda.txt \
  --module_cat_prompt --no_module_visual_guidance \
  --module_rsap_v1 --rsap_topk 3 --rsap_gamma 1.0 \
  --text_adjust True \
  --text_shift --do_shift --per_label \
  --reset_mode domain \
  --corruptions_list original --corruption_severity 5 \
  --ctta_result_dir save_result/rsap_sdr_v4/clean_rsap
```

### 5.3 SDR only (w/o RSAP)

Prediction remains single-prompt. The original multi-description file is used
only by the frozen auxiliary reliability estimator.

```bash
CUDA_VISIBLE_DEVICES=$GPU python ctta_eval_remote.py "$DATA_ROOT" \
  --test_sets loveda -a ViT-B/16 -b 1 --gpu 0 --resolution 448 \
  --lr 1e-4 --tta_steps 1 --num_classes 7 \
  --name_path ./configs/cls_loveda.txt \
  --no_module_cat_prompt --no_module_visual_guidance \
  --text_shift --do_shift --per_label \
  --loss_sdr --lamb_sdr 1.0 --sdr_min_weight 0.5 \
  --reset_mode domain \
  --corruptions_list original --corruption_severity 5 \
  --ctta_result_dir save_result/rsap_sdr_v4/clean_sdr
```

### 5.4 Full: RSAP + SDR

```bash
CUDA_VISIBLE_DEVICES=$GPU python ctta_eval_remote.py "$DATA_ROOT" \
  --test_sets loveda -a ViT-B/16 -b 1 --gpu 0 --resolution 448 \
  --lr 1e-4 --tta_steps 1 --num_classes 7 \
  --name_path ./configs/cls_loveda.txt \
  --module_cat_prompt --no_module_visual_guidance \
  --module_rsap_v1 --rsap_topk 3 --rsap_gamma 1.0 \
  --text_adjust True \
  --text_shift --do_shift --per_label \
  --loss_sdr --lamb_sdr 1.0 --sdr_min_weight 0.5 \
  --reset_mode domain \
  --corruptions_list original --corruption_severity 5 \
  --ctta_result_dir save_result/rsap_sdr_v4/clean_full
```

## 6. Continual CTTA evaluation

Use the same module flags as above, but switch to the continual protocol and
the requested corruption stream.

Example Full run:

```bash
CUDA_VISIBLE_DEVICES=$GPU python ctta_eval_remote.py "$DATA_ROOT" \
  --test_sets loveda -a ViT-B/16 -b 1 --gpu 0 --resolution 448 \
  --lr 1e-4 --tta_steps 1 --num_classes 7 \
  --name_path ./configs/cls_loveda.txt \
  --module_cat_prompt --no_module_visual_guidance \
  --module_rsap_v1 --rsap_topk 3 --rsap_gamma 1.0 \
  --text_adjust True \
  --text_shift --do_shift --per_label \
  --loss_sdr --lamb_sdr 1.0 --sdr_min_weight 0.5 \
  --reset_mode continual \
  --corruptions_list common --corruption_severity 5 \
  --ctta_result_dir save_result/rsap_sdr_v4/ctta_full
```

For SDR-only CTTA, replace the RSAP flags with:

```bash
--no_module_cat_prompt --no_module_visual_guidance
--loss_sdr --lamb_sdr 1.0 --sdr_min_weight 0.5
```

while keeping the original multi-description `--name_path`.

## 7. Reference branch

SDR compares the online model with a frozen source-state copy.

- RSAP off + SDR on:
  - online prediction: single-prompt prediction with continually updated text shift;
  - reference prediction: frozen single-prompt source-state prediction.
- RSAP on + SDR on:
  - online prediction: RSAP prediction with continually updated text shift and alpha;
  - reference prediction: the same RSAP forward structure frozen at source state.

Thus online and reference predictions use the same prediction structure within
an experiment; only the online adaptation parameters change over time.

## 8. Trainable and frozen state

v4 explicitly freezes the complete model first.

The online optimizer then enables only:

- text-shift parameters used by the current prediction path;
- RSAP fusion coefficient `alpha` when RSAP (or legacy Visual Guidance) is active.

The legacy `text_shifter_visual` object remains for compatibility/reset but is
not optimized in v4.

The following remain frozen:

- visual encoder;
- dense feature upsampler;
- source-state reference model;
- multi-description reliability bank;
- reliability statistics themselves.

## 9. Compatibility with v3 Full

For `--module_cat_prompt --module_rsap_v1 --loss_sdr`, the prediction bank and
reliability bank point to the same multi-description embeddings, matching the
v3 Full data flow.

The v4 change is specifically intended to affect the **SDR-only** routing, not
the Full RSAP+SDR algorithm.

Before reusing old Full metrics in the paper, run one short/full verification
to confirm numerical equivalence in the actual CUDA environment.

## 10. Regression checks

Run:

```bash
python -m unittest tests.test_sdr_consistency tests.test_prompt_routing
```

The tests cover:

- `w_min=1` exactly recovers uniform GSC;
- SDR gradients remain on the adapted prediction;
- invalid SDR weight floors are rejected;
- SDR accepts single-prompt prediction while preserving the original
  multi-description reliability bank;
- RSAP still requires Cat-Prompt prediction.
