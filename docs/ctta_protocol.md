# TMPA + DAF-derived CTTA protocol

## 1. Scope

This branch turns TMPA into a strict continual test-time adaptation (CTTA)
evaluator for remote-sensing open-vocabulary segmentation and integrates a set
of DAF-derived stabilization mechanisms.

The project has two controlled stages:

1. **Protocol transfer**: corruption stream + reset semantics + strict sequential evaluation.
2. **Stabilization modules**: Source Consistency, Prompt Feature Consistency, CMAC and Temporal SAFS.

The original TMPA evaluator remains available; the new CTTA entry point is
`ctta_eval_remote.py`.

## 2. Standalone repository

A separate DAF checkout is no longer required.

The relevant DAF/ImageNet-C corruption implementation is vendored under:

```text
ctta/vendor/daf_imagecorruptions/
```

The legacy `--daf_root` argument may still be accepted by older commands for
backward compatibility, but the CTTA corruption path no longer uses it.

DAF-derived source code is distributed with its MIT licence at:

```text
ctta/vendor/DAF_LICENCE
```

### Frost note

The original DAF frost corruption blends the image with bundled photographic
frost textures. This standalone branch uses a deterministic procedural icy
texture instead of those external binary assets while preserving DAF's severity
blend coefficients. Therefore the `frost` domain is protocol-compatible in
name/severity/stream position but is **not pixel-identical** to DAF's original
frost asset implementation. The remaining common corruptions use the vendored
formulas and severity tables.

## 3. Corruption stream

`--corruptions_list common` expands to:

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

Corruption is applied to the raw uint8 RGB image before TMPA resize and CLIP
normalization. NumPy RNG is seeded with the sample index and restored after each
corruption so each sample has a deterministic corruption realization.

Severity is configurable from 1 to 5; severity 5 is the default CTTA setting.

## 4. Reset modes

- `source`: no test-time update.
- `episodic`: reset TMPA tunable state and optimizer before every sample.
- `domain`: retain adaptation within one corruption domain and reset at domain boundaries.
- `continual`: reset once before the stream and retain adaptation across all samples and domains.

## 5. Strict temporal stream

`ctta_eval_remote.py` requires:

```text
batch_size = 1
WORLD_SIZE = 1
```

This prevents DDP from partitioning a single temporal stream across ranks. Use
multiple GPUs for independent datasets, seeds or ablations rather than for one
shared CTTA stream.

## 6. Phase-2 modules

### Source Prediction Consistency

A frozen source TMPA model anchors the adapted prediction using symmetric KL.

```bash
--loss_src_cons --lamb_src_cons 1.0
```

### Prompt Feature Consistency

DAF's visual feature consistency is not copied literally because TMPA freezes
its visual encoder during adaptation. The corresponding regularizer is applied
to the shifted prompt/text features that TMPA actually updates.

```bash
--loss_prompt_feat_cons \
--lamb_prompt_feat_cons 1.0 \
--prompt_feat_cons_type cosine
```

Supported distances: `cosine`, `l2`.

### CMAC

The frozen source prediction assigns an anchor class per pixel. The loss only
penalizes harmful drift:

- source-anchor probability decreases;
- source-unassigned class probability increases.

```bash
--loss_cmac --lamb_cmac 1.0
```

### Temporal SAFS

DAF SAFS is a selector rather than another loss. DAF computes its threshold from
feature-shift statistics across a batch, but strict CTTA here uses `batch_size=1`.
The selector is therefore converted to a temporal rolling-history gate.

For each sample, adapted/source prediction drift is measured and compared with:

```text
threshold = history_mean - alpha_safs * history_std
```

If the sample does not pass the gate, its optimizer step is skipped.

```bash
--module_safs \
--alpha_safs 0.5 \
--safs_window 32 \
--safs_warmup 8
```

`continual` retains SAFS history across corruption domains. `domain` clears SAFS
history when a new corruption domain begins.

## 7. Recommended ablation matrix

Keep dataset, seed, severity, corruption order, learning rate and TTA steps fixed.

1. TMPA-Continual
2. TMPA-Continual + Source Consistency
3. TMPA-Continual + Prompt Feature Consistency
4. TMPA-Continual + CMAC
5. TMPA-Continual + Temporal SAFS
6. TMPA-Continual + Source Consistency + CMAC
7. TMPA-Continual + CMAC + SAFS
8. TMPA-Continual + all implemented stabilizers

## 8. Minimal smoke test

Run one corruption first:

```bash
CUDA_VISIBLE_DEVICES=0 python ctta_eval_remote.py /path/to/data \
  --test_sets loveda \
  -a ViT-B/16 \
  -b 1 \
  --gpu 0 \
  --lr 1e-4 \
  --tta_steps 1 \
  --num_classes 7 \
  --name_path ./configs/cls_loveda.txt \
  --text_shift --do_shift --per_label \
  --text_adjust True \
  --reset_mode continual \
  --corruptions_list gaussian_noise \
  --corruption_severity 5
```

After this succeeds, use:

```bash
--corruptions_list common
```

for the full 15-domain stream.

## 9. Outputs

The evaluator records:

- per-domain segmentation metrics;
- whole-stream metrics;
- cumulative stream-progress metrics;
- enabled stabilization modules and weights;
- adaptation-loss summaries;
- SAFS keep/filter rate, shift score and threshold statistics when enabled.

Result filenames include the enabled module tags so independent ablations do not
overwrite the baseline.

## 10. External requirements that remain

The repository is now standalone **relative to DAF**, but it still requires the
normal TMPA assets/environment:

- remote-sensing datasets;
- TMPA/CLIP/OpenCLIP model assets;
- SimFeatUp checkpoint/weights;
- CUDA/PyTorch and the dependencies in `requirements.txt`.

These are TMPA model/data dependencies, not dependencies on the DAF repository.
