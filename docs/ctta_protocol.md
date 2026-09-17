# TMPA + DAF CTTA evaluation protocol

## Scope

Phase 1 changes the **evaluation protocol only**. It keeps TMPA's remote-sensing
datasets, prompts, model, SimFeatUp inference and entropy-based adaptation, while
reusing DAF's corruption implementation and continual reset semantics.

No DAF adaptation loss/module is enabled in Phase 1.

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

## Example

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

## Phase 2

DAF source consistency, feature consistency, CMAC, and SAFS should be migrated
one at a time under `ctta/modules/`, keeping this Phase-1 protocol fixed. This
allows each stabilization module's contribution to be measured against the
same TMPA-Continual baseline.
