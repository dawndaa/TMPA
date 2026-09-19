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
