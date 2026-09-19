# RSAP + SDR v2

This branch develops the paper-facing RSAP + SDR method on top of `feature/rsap-v1`.

## Method scope

The branch intentionally keeps the already effective GSC stabilization backbone and
changes only how its pixel importance is assigned.

### RSAP

RSAP replaces legacy TMPA Visual Guidance when enabled together with:

    --module_rsap_v1 --no_module_visual_guidance --text_adjust True

It keeps Cat-Prompt multi-description semantics, estimates class support and
prompt disagreement using the frozen Cat-Prompt embeddings, and computes:

    R_(i,c) = mean_k p_(i,c)^(k) * exp(-gamma * var_k p_(i,c)^(k))

For each predicted class, top-k reliable tokens form a reliability-weighted visual
prototype. Class reliability gates text/visual calibration.

The reliability estimator uses frozen Cat-Prompt embeddings rather than the
continually shifted text state so the gate cannot reinforce its own prompt drift.

### SDR

SDR preserves the original symmetric-KL source prediction consistency (GSC):

    d_i = 0.5 * [ KL(p_i^a || p_i^s) + KL(p_i^s || p_i^a) ]

and changes only its pixel weighting:

    w_i = w_min + (1 - w_min) * (1 - r_i)

    L_SDR = sum_i w_i d_i / sum_i w_i

where r_i is the detached RSAP reliability map.

Properties:

- low reliability -> nearly full source anchoring;
- high reliability -> weaker source anchoring;
- w_min > 0 prevents reliable pixels from losing source regularization entirely;
- normalizing by sum(w_i) keeps SDR on a comparable scale to the original GSC;
- setting w_min = 1 exactly recovers uniform GSC.

Default:

    --sdr_min_weight 0.5

## Important ablations

Uniform GSC:

    --loss_src_cons

RSAP only:

    --module_rsap_v1 --no_module_visual_guidance --text_adjust True

RSAP + SDR:

    --module_rsap_v1 --no_module_visual_guidance --text_adjust True --loss_sdr

Do not combine `--loss_src_cons` and `--loss_sdr`; SDR already contains GSC.

CMAC/DSA remains available in the repository for earlier DAF-derived experiments,
but it is not part of the proposed RSAP + SDR v2 method.

## Recommended first experiment

Use the same LoveDA Urban clean protocol and hyperparameters as the existing
M1/GSC run:

1. Cat-Prompt + Text-shift + uniform GSC.
2. RSAP + Text-shift + uniform GSC.
3. RSAP + Text-shift + SDR, w_min=0.5.
4. If needed, sweep only w_min in {0.3, 0.5, 0.7, 1.0}.

The w_min=1.0 case is a regression control and should match the uniform GSC loss
up to floating-point differences when the prediction paths are otherwise identical.


## LoveDA Urban clean: exact ablation commands

The following five runs correspond to the requested main-module ablation.
They assume that `DATA_ROOT` is the same LoveDA root used by the existing
LoveDA Urban clean experiment and resolves to the same 677 validation images.

All clean adaptation runs use one `original` domain and `reset_mode=domain`
to match the existing clean protocol. With one domain, parameters are retained
across all 677 images. Use `reset_mode=continual` later for multi-corruption
CTTA streams.

Set once:

```bash
export DATA_ROOT=/path/to/data
export GPU=0
```

Shared arguments for all runs:

```text
--test_sets loveda
-a ViT-B/16
-b 1
--gpu <GPU>
--resolution 448
--lr 1e-4
--tta_steps 1
--num_classes 7
--name_path ./configs/cls_loveda.txt
--corruptions_list original
--corruption_severity 5
```

### A0. SegEarth-OV Source

No Cat-Prompt, no legacy Visual Guidance, and no test-time update.

```bash
CUDA_VISIBLE_DEVICES=$GPU python ctta_eval_remote.py "$DATA_ROOT" \
  --test_sets loveda -a ViT-B/16 -b 1 --gpu 0 --resolution 448 \
  --lr 1e-4 --tta_steps 1 --num_classes 7 \
  --name_path ./configs/cls_loveda.txt \
  --no_module_cat_prompt --no_module_visual_guidance \
  --reset_mode source \
  --corruptions_list original --corruption_severity 5 \
  --ctta_result_dir save_result/rsap_sdr_v2/A0_source
```

### A1. SegEarth-OV Source + Text-shift

No Cat-Prompt and no legacy Visual Guidance. Only the existing text-shift state
is optimized and retained across the clean domain.

```bash
CUDA_VISIBLE_DEVICES=$GPU python ctta_eval_remote.py "$DATA_ROOT" \
  --test_sets loveda -a ViT-B/16 -b 1 --gpu 0 --resolution 448 \
  --lr 1e-4 --tta_steps 1 --num_classes 7 \
  --name_path ./configs/cls_loveda.txt \
  --no_module_cat_prompt --no_module_visual_guidance \
  --text_shift --do_shift --per_label \
  --reset_mode domain \
  --corruptions_list original --corruption_severity 5 \
  --ctta_result_dir save_result/rsap_sdr_v2/A1_textshift
```

### A2. SegEarth-OV Source + Text-shift + GSC

This is the uniform GSC baseline. Use `--loss_src_cons`, not `--loss_sdr`.

```bash
CUDA_VISIBLE_DEVICES=$GPU python ctta_eval_remote.py "$DATA_ROOT" \
  --test_sets loveda -a ViT-B/16 -b 1 --gpu 0 --resolution 448 \
  --lr 1e-4 --tta_steps 1 --num_classes 7 \
  --name_path ./configs/cls_loveda.txt \
  --no_module_cat_prompt --no_module_visual_guidance \
  --text_shift --do_shift --per_label \
  --loss_src_cons --lamb_src_cons 1.0 \
  --reset_mode domain \
  --corruptions_list original --corruption_severity 5 \
  --ctta_result_dir save_result/rsap_sdr_v2/A2_textshift_gsc
```

### A3. SegEarth-OV Source + Text-shift + SDR

This isolates the modified GSC without enabling RSAP prompt calibration.
SDR still needs Cat-Prompt because multi-prompt consensus is the reliability
estimator, but `--module_rsap_v1` remains off.

```bash
CUDA_VISIBLE_DEVICES=$GPU python ctta_eval_remote.py "$DATA_ROOT" \
  --test_sets loveda -a ViT-B/16 -b 1 --gpu 0 --resolution 448 \
  --lr 1e-4 --tta_steps 1 --num_classes 7 \
  --name_path ./configs/cls_loveda.txt \
  --module_cat_prompt --no_module_visual_guidance \
  --text_shift --do_shift --per_label \
  --loss_sdr --lamb_sdr 1.0 --sdr_min_weight 0.5 \
  --reset_mode domain \
  --corruptions_list original --corruption_severity 5 \
  --ctta_result_dir save_result/rsap_sdr_v2/A3_textshift_sdr
```

### A4. SegEarth-OV Source + Text-shift + RSAP

RSAP uses Cat-Prompt as its multi-description semantic input and replaces legacy
TMPA Visual Guidance. Therefore Cat-Prompt is explicitly enabled while legacy
Visual Guidance is explicitly disabled.

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
  --ctta_result_dir save_result/rsap_sdr_v2/A4_textshift_rsap
```

### A5. SegEarth-OV Source + Text-shift + RSAP + GSC

This combines RSAP with the original uniform GSC. It is useful for separating
the effect of RSAP from the later reliability-guided SDR modification.

```bash
CUDA_VISIBLE_DEVICES=$GPU python ctta_eval_remote.py "$DATA_ROOT" \
  --test_sets loveda -a ViT-B/16 -b 1 --gpu 0 --resolution 448 \
  --lr 1e-4 --tta_steps 1 --num_classes 7 \
  --name_path ./configs/cls_loveda.txt \
  --module_cat_prompt --no_module_visual_guidance \
  --module_rsap_v1 --rsap_topk 3 --rsap_gamma 1.0 \
  --text_adjust True \
  --text_shift --do_shift --per_label \
  --loss_src_cons --lamb_src_cons 1.0 \
  --reset_mode domain \
  --corruptions_list original --corruption_severity 5 \
  --ctta_result_dir save_result/rsap_sdr_v2/A5_textshift_rsap_gsc
```

### A6. RSAP + SDR (full v2 method)

A5 still uses uniform GSC. The proposed v2 SDR is the reliability-guided GSC.
To evaluate the actual RSAP + SDR full method, replace `--loss_src_cons` with
`--loss_sdr`:

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
  --ctta_result_dir save_result/rsap_sdr_v2/A6_textshift_rsap_sdr
```

Do not combine `--loss_src_cons` and `--loss_sdr`; SDR already contains GSC.

### Interpretation caveat: Cat-Prompt control

A0-A2 intentionally disable Cat-Prompt, whereas A3-A5 require Cat-Prompt because
multi-prompt consensus is the input to RSAP. Therefore the difference A1 -> A3
contains both the availability of multiple category descriptions and the new RSAP
reliability mechanism.

For strict mechanism-level attribution, note that standalone SDR also requires Cat-Prompt.
Therefore A2 -> A3 changes both the prompt source (single prompt -> Cat-Prompt) and the GSC weighting.
If exact isolation is needed, add these controls:

```text
Text-shift + Cat-Prompt
Text-shift + Cat-Prompt + uniform GSC
```

Then compare Cat-Prompt + GSC against Cat-Prompt + SDR to isolate reliability-guided weighting,
and Cat-Prompt against Cat-Prompt + RSAP to isolate RSAP calibration.

For a strict mechanism-level RSAP attribution, add one inexpensive control:

```text
Text-shift + Cat-Prompt, with --no_module_visual_guidance and without --module_rsap_v1
```

This control is not mandatory for the five-row main-module ablation, but it is
recommended if space or reviewer scrutiny requires isolating the gain from RSAP
beyond Cat-Prompt itself.
