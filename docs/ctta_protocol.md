# TMPA + DAF CTTA evaluation protocol

## Scope

The project is split into two controlled stages.

**Phase 1** changes the evaluation protocol only. It keeps TMPA's remote-sensing
datasets, prompts, model, SimFeatUp inference and entropy-based adaptation, while
reusing DAF's corruption implementation and continual reset semantics.

**Phase 2** keeps that protocol fixed and adds DAF-derived stabilization modules
one at a time. This makes every gain attributable to an explicit module rather
than to a changed stream or corruption setup.

## Corruption stream

`--corruptions_list common` expands to DAF's 15 common corruptions, in order:

1. gaussian_noise
2. shot_noise
3. impulse_noise
4. defocus_blur
5. glass_blur
6. motion_blur
7. zoom_blur
8. snow
9. frost
10. fog
11. brightness
12. contrast
13. elastic_transform
14. pixelate
15. jpeg_compression

Corruption is applied on the raw uint8 RGB image **before** TMPA resize and CLIP
normalization. The NumPy RNG is seeded with the sample index and restored after
each corruption, matching DAF's `CorruptTransform` behavior.

DAF's main code uses severity 5; this implementation supports severities 1-5.

## Reset modes

- `source`: no test-time update.
- `episodic`: reset TMPA tunable state and optimizer before every sample.
- `domain`: retain state within one corruption domain, reset when the corruption changes.
- `continual`: reset once before the first sample and retain state across all samples and corruption domains.

## Strict CTTA stream

`ctta_eval_remote.py` requires `batch_size=1` and `WORLD_SIZE=1`. This avoids a
protocol ambiguity in the original TMPA DDP evaluator, where different ranks
consume different samples and synchronize gradients, effectively producing a
multi-sample batch-CTTA update rather than a single temporal stream.

Use multiple GPUs by running independent datasets/seeds on different GPUs.

## DAF dependency

The evaluator loads `DAF/utils/imagecorruptions` directly from a local DAF
checkout. This preserves DAF's exact formulas and frost assets without copying
binary resources into TMPA during Phase 1.

Pass `--daf_root /path/to/DAF` or set `DAF_ROOT`.

## Phase-1 baseline example

```bash
CUDA_VISIBLE_DEVICES=0 python ctta_eval_remote.py /path/to/data \
  --test_sets loveda \
  -a ViT-B/16 \
  -b 1 \
  --gpu 0 \
  --lr 1e-4 \
  --tta_steps 3 \
  --num_classes 7 \
  --name_path ./configs/cls_loveda.txt \
  --text_shift --do_shift --per_label \
  --dataset_name loveda \
  --text_adjust True \
  --reset_mode continual \
  --corruptions_list common \
  --corruption_severity 5 \
  --daf_root /path/to/DAF
```

Run the same command with `source`, `episodic`, `domain`, and `continual` to
produce the Phase-1 comparison matrix.

## Phase-2 module 1: source prediction consistency

DAF computes a symmetric KL penalty between the adapted segmentation prediction
and a frozen source model prediction. TMPA exposes post-processed class
probability maps rather than DAF's raw class logits, so the maps are normalized
per pixel and the same symmetric-KL principle is applied.

Enable it with:

```bash
--loss_src_cons --lamb_src_cons 1.0
```

A frozen source copy is created only when a Phase-2 source-anchored module is
enabled. The ordinary Phase-1 path does not allocate the extra source model.

## Phase-2 module 2: prompt/text feature consistency

DAF's original feature consistency compares adapted **visual** features with
frozen source visual features. That loss cannot be copied literally into TMPA:
TMPA freezes the visual encoder and SimFeatUp during adaptation, so adapted and
source visual features are the same and the loss would be identically zero.

The TMPA-space counterpart therefore regularizes the feature that actually
changes during adaptation: the shifted prompt/text embedding.

Enable it with:

```bash
--loss_prompt_feat_cons \
--lamb_prompt_feat_cons 1.0 \
--prompt_feat_cons_type cosine
```

`cosine` and `l2` are supported. The option requires `--text_shift` so a
meaningful adaptive prompt feature exists.

## Phase-2 module 3: CMAC directional source anchoring

DAF CMAC uses the frozen source visual feature to assign an anchor class to each
patch. It then applies two directional hinge terms:

1. penalize the adapted representation if similarity to its source-assigned
   class decreases (`loss_away`);
2. penalize the adapted representation if similarity to any source-unassigned
   class increases (`loss_toward`).

TMPA does not adapt its visual encoder, so the DAF feature-space implementation
cannot be copied literally. The transferred version applies the same directional
principle to TMPA's pixel-wise class probabilities:

- the frozen source prediction assigns the anchor class per pixel;
- adapted anchor probability is not allowed to decrease without penalty;
- adapted non-anchor probabilities are not allowed to increase without penalty.

This is deliberately different from symmetric source consistency: CMAC only
penalizes harmful drift directions and does not penalize movement that strengthens
the source-assigned class or suppresses alternatives.

Enable it with:

```bash
--loss_cmac --lamb_cmac 1.0
```

## Phase-2 ablation matrix

Keep the same dataset, corruption order, severity, seed, learning rate and TTA
steps for every row:

1. TMPA-Continual
2. TMPA-Continual + Source Consistency
3. TMPA-Continual + Prompt Feature Consistency
4. TMPA-Continual + CMAC
5. TMPA-Continual + Source Consistency + Prompt Feature Consistency
6. TMPA-Continual + Source Consistency + CMAC
7. TMPA-Continual + all implemented stabilizers

Example for source consistency:

```bash
CUDA_VISIBLE_DEVICES=0 python ctta_eval_remote.py /path/to/data \
  --test_sets loveda \
  -a ViT-B/16 -b 1 --gpu 0 \
  --lr 1e-4 --tta_steps 3 \
  --num_classes 7 \
  --name_path ./configs/cls_loveda.txt \
  --text_shift --do_shift --per_label \
  --text_adjust True \
  --reset_mode continual \
  --corruptions_list common \
  --corruption_severity 5 \
  --daf_root /path/to/DAF \
  --loss_src_cons --lamb_src_cons 1.0
```

Example for CMAC:

```bash
CUDA_VISIBLE_DEVICES=0 python ctta_eval_remote.py /path/to/data \
  --test_sets loveda \
  -a ViT-B/16 -b 1 --gpu 0 \
  --lr 1e-4 --tta_steps 3 \
  --num_classes 7 \
  --name_path ./configs/cls_loveda.txt \
  --text_shift --do_shift --per_label \
  --text_adjust True \
  --reset_mode continual \
  --corruptions_list common \
  --corruption_severity 5 \
  --daf_root /path/to/DAF \
  --loss_cmac --lamb_cmac 1.0
```

Result filenames include enabled module names and weights, so Phase-2 runs do
not overwrite the Phase-1 baseline.

## Next module

SAFS should be integrated after the three current stabilizers are evaluated
independently. Its implementation should preserve the same Phase-1 stream and
be treated as an update/sample-filtering mechanism rather than silently changing
the corruption protocol.
