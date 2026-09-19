# RSAP-v1: Reliability-aware Scene-Adaptive Prompting

This branch adds an opt-in first version of Reliability-aware Scene-Adaptive
Prompting (RSAP) on top of the existing TMPA + CTTA implementation.

## Scope

RSAP-v1 intentionally implements only the two lowest-risk changes:

1. Multi-prompt consensus visual mining.
2. Reliability-gated text/visual fusion.

The proposed scene-aware prompt aggregation stage is intentionally deferred.
The default code path remains the original TMPA implementation.

## Motivation

TMPA selects low-entropy visual tokens and injects their visual features into
text prompts. In a continual test-time stream, current predictions can drift,
so confidence alone can become unreliable. RSAP-v1 asks whether multiple text
descriptions of the same semantic class agree before using the current visual
evidence to calibrate that class.

## Consensus and reliability

Prompt files may have unequal numbers of descriptions per class. Therefore,
RSAP-v1 does not align the k-th prompt across all classes.

For each class c:

- Build a mean text prototype for every class.
- Replace class c's mean prototype by each individual prompt t_(c,k) in turn.
- Compute an independent C-way class probability p_(i,c)^(k) for each visual
  token i.
- Aggregate all prompts belonging to class c:

    C_(i,c) = mean_k p_(i,c)^(k)

    U_(i,c) = var_k p_(i,c)^(k)

    R_(i,c) = C_(i,c) * exp(-gamma * U_(i,c))

The predicted class is argmax_c C_(i,c). For every predicted class, RSAP-v1
keeps the top-k visual tokens according to R_(i,c) and builds a
reliability-weighted visual prototype.

Reliability is detached from autograd. It acts as a selection/gating statistic
and cannot be directly manipulated by the entropy-minimization objective.

## Reliability-gated calibration

For each class c, let R_c be the mean reliability of its selected visual tokens.
For every prompt q belonging to class c:

    beta_q = clip(alpha_q * R_c, 0, 1)

    t_tilde_q = (1 - beta_q) * t_q + beta_q * v_c

alpha_q remains the learnable prompt-calibration parameter, while R_c is
sample- and class-dependent. Clipping beta_q to [0, 1] guarantees a valid
interpolation coefficient even if continual optimization moves alpha_q outside
the unit interval.

## Flags

RSAP-v1 is disabled by default. Enable it with:

    --module_rsap_v1

Optional parameters:

    --rsap_topk 3
    --rsap_gamma 1.0

RSAP-v1 requires:

    --text_adjust True
    --module_visual_guidance
    Cat-Prompt enabled (default)

## LoveDA CTTA example

Baseline TMPA CTTA:

    CUDA_VISIBLE_DEVICES=0 python ctta_eval_remote.py /path/to/data \
      --test_sets loveda -a ViT-B/16 -b 1 --gpu 0 \
      --lr 1e-4 --tta_steps 1 --num_classes 7 \
      --name_path ./configs/cls_loveda.txt \
      --text_shift --do_shift --per_label --text_adjust True \
      --reset_mode continual --corruptions_list common \
      --corruption_severity 5

RSAP-v1:

    <same command> --module_rsap_v1

RSAP-v1 + Source Consistency:

    <same command> --module_rsap_v1 --loss_src_cons

RSAP-v1 + CMAC:

    <same command> --module_rsap_v1 --loss_cmac

RSAP-v1 + Source Consistency + CMAC:

    <same command> --module_rsap_v1 --loss_src_cons --loss_cmac

## Recommended first ablation

Compare under exactly the same stream and hyperparameters:

- TMPA CTTA
- TMPA CTTA + RSAP-v1
- TMPA CTTA + Source Consistency + CMAC
- TMPA CTTA + RSAP-v1 + Source Consistency + CMAC

Then split RSAP-v1 itself if it is effective:

- consensus visual mining only
- reliability gate only
- both

The current branch implements the combined v1 path first to minimize code churn.
