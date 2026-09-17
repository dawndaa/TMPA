# Standalone TMPA-CTTA

This branch is self-contained with respect to the DAF integration. You do **not** need a separate DAF checkout.

## What is vendored

- DAF/ImageNet-C corruption interface under `ctta/vendor/daf_imagecorruptions/`
- common corruption formulas/severity tables used by the CTTA evaluator
- DAF-derived Source Consistency / Prompt Feature Consistency / CMAC / Temporal SAFS modules under `ctta/modules/`
- DAF MIT licence under `ctta/vendor/DAF_LICENCE`

The original DAF frost corruption uses bundled photographic frost textures. To keep this repository standalone without carrying those external binary assets, this branch uses a deterministic procedural frost texture while preserving DAF's severity blending coefficients. The other common corruption implementations use the vendored formulas and severity tables.

## Installation

```bash
git clone https://github.com/dawndaa/TMPA.git
cd TMPA
git switch --track origin/feature/ctta-daf-integration
pip install -r requirements.txt
```

The existing `requirements.txt` already contains OpenCV, Pillow, SciPy and scikit-image. `environment.yaml` has also been updated with the corruption runtime dependencies.

A separate DAF repository is no longer required. The legacy `--daf_root` argument is accepted for backward compatibility but ignored.

## Minimal smoke test

Start with one corruption rather than the full stream:

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

After that succeeds, replace `gaussian_noise` with `common` to run the 15-domain stream.

## DAF-derived module examples

Source consistency:

```bash
--loss_src_cons --lamb_src_cons 1.0
```

Prompt feature consistency:

```bash
--loss_prompt_feat_cons --lamb_prompt_feat_cons 1.0 --prompt_feat_cons_type cosine
```

CMAC:

```bash
--loss_cmac --lamb_cmac 1.0
```

Temporal SAFS:

```bash
--module_safs --alpha_safs 0.5 --safs_window 32 --safs_warmup 8
```

## Still external

The repository is standalone relative to DAF, but TMPA itself still requires the original model/checkpoint/data assets used by TMPA/SimFeatUp and the remote-sensing datasets. Those are model/data dependencies, not DAF code dependencies.
