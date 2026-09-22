import argparse
import os
import tempfile
from pathlib import Path


def _build_single_prompt_file(name_path: str) -> str:
    """Create a per-process single-prompt view of an existing Cat-Prompt file.

    Each non-empty source line keeps only the text before the first ``|``.
    This preserves the dataset's canonical category wording while removing all
    Cat-Prompt descriptions. A temporary file is used so the original prompt
    configuration is never modified, including under multi-process evaluation.
    """
    source = Path(name_path)
    if not source.is_file():
        raise FileNotFoundError(
            f'Cannot disable Cat-Prompt because --name_path does not exist: {name_path}'
        )

    category_prompts = []
    for line in source.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        category_prompt = line.split('|', 1)[0].strip()
        if category_prompt:
            category_prompts.append(category_prompt)

    if not category_prompts:
        raise ValueError(f'No category prompts found in {name_path}.')

    fd, output_path = tempfile.mkstemp(
        prefix=f'tmpa_{source.stem}_single_{os.getpid()}_',
        suffix='.txt',
    )
    with os.fdopen(fd, 'w', encoding='utf-8') as f:
        f.write('\n'.join(category_prompts) + '\n')
    return output_path


def parse_args():
    parser = argparse.ArgumentParser(description='Test-time Prompt Tuning')
    parser.add_argument('data', metavar='DIR', help='path to dataset root')
    parser.add_argument('--test_sets', type=str, default='A/R/V/K/I', help='test dataset (multiple datasets split by slash)')
    parser.add_argument('--dataset_mode', type=str, default='test', help='which split to use: train/val/test')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='RN50')
    parser.add_argument('--resolution', default=448, type=int, help='CLIP image resolution')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-b', '--batch_size', default=64, type=int, metavar='N')
    parser.add_argument('--lr', '--learning-rate', default=5e-3, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('-p', '--print-freq', default=200, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--gpu', default=0, type=int,
                        help='GPU id to use.')
    parser.add_argument('--tpt', action='store_true', default=False, help='run test-time prompt tuning')
    parser.add_argument('--selection_p', default=6, type=float, help='confidence selection percentile')
    parser.add_argument('--vis_feat', default=None)
    parser.add_argument('--tta_steps', default=1, type=int, help='test-time-adapt steps')
    parser.add_argument('--n_ctx', default=4, type=int, help='number of tunable text tokens')
    parser.add_argument('--ctx_init', default=None, type=str, help='init tunable text prompts')
    parser.add_argument('--cocoop', action='store_true', default=False, help="use cocoop's output as prompt initialization")
    parser.add_argument('--load', default=None, type=str, help='path to a pre-trained coop/cocoop')
    parser.add_argument('--seed', type=int, default=0)

    # CTTA / DAF evaluation protocol
    parser.add_argument(
        '--reset_mode',
        type=str,
        default='episodic',
        choices=('source', 'episodic', 'domain', 'continual'),
        help=(
            'TTA state protocol: source=no adaptation; episodic=reset before every sample; '
            'domain=reset at every corruption domain; continual=never reset inside the stream.'
        ),
    )
    parser.add_argument(
        '--corruptions_list',
        nargs='+',
        default=['original'],
        help=(
            "Corruption domains in stream order. Use 'common' for the 15 DAF/ImageNet-C corruptions, "
            "or specify names explicitly. 'original' evaluates clean images."
        ),
    )
    parser.add_argument(
        '--corruption_severity',
        type=int,
        default=5,
        choices=(1, 2, 3, 4, 5),
        help='DAF corruption severity. DAF main experiments use severity 5.',
    )
    parser.add_argument(
        '--daf_root',
        type=str,
        default=None,
        help=(
            'Deprecated compatibility option. DAF corruption code is vendored inside TMPA, '
            'so no external DAF checkout is required and this value is ignored.'
        ),
    )
    parser.add_argument(
        '--ctta_progress',
        nargs='+',
        type=float,
        default=[0.1, 0.2, 0.4, 0.8, 1.0],
        help='Fractions of the continual stream at which cumulative metrics are reported.',
    )
    parser.add_argument(
        '--ctta_result_dir',
        type=str,
        default='save_result/ctta',
        help='Directory for CTTA JSON result files.',
    )

    # Phase-2 DAF-derived stabilization modules
    parser.add_argument(
        '--loss_src_cons',
        action='store_true',
        help='Enable DAF-style source prediction consistency against a frozen TMPA source model.',
    )
    parser.add_argument(
        '--lamb_src_cons',
        type=float,
        default=1.0,
        help='Weight for source prediction consistency.',
    )
    parser.add_argument(
        '--loss_sdr',
        action='store_true',
        help=(
            'Enable reliability-guided semantic drift regulation (SDR): retain the '
            'original symmetric-KL source consistency, but weight pixels using a '
            'detached multi-description reliability map with a protected minimum anchor '
            'weight. When Cat-Prompt/RSAP prediction is disabled, SDR uses the original '
            'multi-description prompt file only as a frozen auxiliary reliability bank.'
        ),
    )
    parser.add_argument(
        '--lamb_sdr',
        type=float,
        default=1.0,
        help='Weight for reliability-guided SDR / GSC.',
    )
    parser.add_argument(
        '--sdr_min_weight',
        type=float,
        default=0.5,
        help=(
            'Minimum per-pixel source-consistency weight in SDR. 1.0 exactly recovers '
            'uniform GSC; lower values allow reliable target evidence more plasticity.'
        ),
    )
    parser.add_argument(
        '--loss_prompt_feat_cons',
        action='store_true',
        help=(
            'Enable prompt/text feature consistency. This is the TMPA-space counterpart '
            'to DAF visual feature consistency because TMPA freezes its visual encoder.'
        ),
    )
    parser.add_argument(
        '--lamb_prompt_feat_cons',
        type=float,
        default=1.0,
        help='Weight for prompt/text feature consistency.',
    )
    parser.add_argument(
        '--prompt_feat_cons_type',
        type=str,
        default='cosine',
        choices=('cosine', 'l2'),
        help='Distance used by prompt/text feature consistency.',
    )
    parser.add_argument(
        '--loss_cmac',
        action='store_true',
        help=(
            'Enable DAF-inspired CMAC directional source anchoring. The frozen source '
            'prediction assigns each pixel an anchor class; harmful drift away from the '
            'anchor or toward non-anchor classes is penalized.'
        ),
    )
    parser.add_argument(
        '--lamb_cmac',
        type=float,
        default=1.0,
        help='Weight for the CMAC directional source-anchor loss.',
    )
    parser.add_argument(
        '--loss_div',
        action='store_true',
        help=(
            'Enable DAF-style class diversity regularization. The loss maximizes the entropy '
            'of the marginal class distribution across pixels to discourage class collapse.'
        ),
    )
    parser.add_argument(
        '--lamb_div',
        type=float,
        default=1.0,
        help='Weight for the DAF-style class diversity loss.',
    )
    parser.add_argument(
        '--module_safs',
        action='store_true',
        help=(
            'Enable temporal SAFS update gating. The original DAF batch-level selector is '
            'adapted to strict batch-size-one CTTA using recent stream history.'
        ),
    )
    parser.add_argument('--alpha_safs', type=float, default=0.5,
                        help='SAFS threshold coefficient in mean - alpha * std.')
    parser.add_argument('--safs_window', type=int, default=32,
                        help='Number of recent shift scores used by temporal SAFS.')
    parser.add_argument('--safs_warmup', type=int, default=8,
                        help='Number of initial scores kept before temporal SAFS starts filtering.')

    # TMPA main-module ablations. Defaults preserve the original full TMPA path.
    cat_prompt_group = parser.add_mutually_exclusive_group()
    cat_prompt_group.add_argument(
        '--module_cat_prompt',
        dest='module_cat_prompt',
        action='store_true',
        help='Enable TMPA Cat-Prompt multi-description prompts (default).',
    )
    cat_prompt_group.add_argument(
        '--no_module_cat_prompt',
        dest='module_cat_prompt',
        action='store_false',
        help='Disable Cat-Prompt and use exactly one naive category-name prompt per class.',
    )

    visual_guidance_group = parser.add_mutually_exclusive_group()
    visual_guidance_group.add_argument(
        '--module_visual_guidance',
        dest='module_visual_guidance',
        action='store_true',
        help='Enable TMPA visual-guided prompt adjustment when --text_adjust True (default).',
    )
    visual_guidance_group.add_argument(
        '--no_module_visual_guidance',
        dest='module_visual_guidance',
        action='store_false',
        help=(
            'Disable only TMPA visual-guided prompt adjustment. Test-time optimization '
            'remains controlled by the existing reset_mode/tta_steps/text_shift logic.'
        ),
    )
    # Backward-compatible aliases for commands produced before VGTA was split.
    # They now control visual guidance only; they no longer disable TTA updates.
    visual_guidance_group.add_argument(
        '--module_vgta',
        dest='module_visual_guidance',
        action='store_true',
        help=argparse.SUPPRESS,
    )
    visual_guidance_group.add_argument(
        '--no_module_vgta',
        dest='module_visual_guidance',
        action='store_false',
        help=argparse.SUPPRESS,
    )
    parser.set_defaults(module_cat_prompt=True, module_visual_guidance=True)

    # Reliability-aware Scene-Adaptive Prompting (RSAP-v1).
    # v1 intentionally keeps TMPA's existing multi-prompt prediction aggregation
    # and changes only visual evidence mining + text/visual fusion.
    parser.add_argument(
        '--module_rsap_v1',
        action='store_true',
        help=(
            'Enable RSAP-v1 as an alternative to legacy Visual Guidance: use frozen '
            'multi-prompt consensus to mine reliable visual prototypes and gate '
            'text/visual fusion by class reliability. Disabled by default.'
        ),
    )
    parser.add_argument(
        '--no_reliability',
        action='store_true',
        help=(
            'Full w/o reliability ablation (requires RSAP + SDR): keep consensus '
            'class assignment, sample K tokens at fixed evenly spaced candidate '
            'positions, average them uniformly, use alpha without reliability '
            'gating, and replace SDR pixel weighting with uniform symmetric KL.'
        ),
    )
    parser.add_argument(
        '--rsap_gamma',
        type=float,
        default=1.0,
        help='Penalty strength for prompt disagreement in RSAP reliability.',
    )
    parser.add_argument(
        '--rsap_topk',
        type=int,
        default=3,
        help='Number of highest-reliability visual tokens per predicted class in RSAP-v1.',
    )

    # TPS
    parser.add_argument('--img_aug', action="store_true")
    parser.add_argument('--with_concepts', action="store_true")
    parser.add_argument('--with_templates', action="store_true")
    parser.add_argument('--with_coop', action="store_true")
    parser.add_argument('--concept_type', type=str, default='gpt4', help='concepts to choose from')
    parser.add_argument('--logname', type=str)
    parser.add_argument('--loss_prompt', default=False)
    parser.add_argument('--text_adjust', default=False)
    parser.add_argument('--alpha', default=0.02, type=float)
    parser.add_argument('--tps_entropy_scale', default=1.0, type=float)

    parser.add_argument('--prob_thd', default=0.1, type=float)
    parser.add_argument('--cls_token_lambda', default=-0.3, type=float)
    parser.add_argument('--logit_scale', default=-0.3, type=float)
    parser.add_argument('--logit_weight', default=-0.3, type=float)
    parser.add_argument('--bg_idx', default=-0.3, type=float)
    parser.add_argument('--prompt_logit_scale', default=1.0, type=float)
    parser.add_argument('--prompt_logit_weight', default=1.0, type=float)

    parser.add_argument('--init_concepts', action="store_true")
    parser.add_argument('--per_label', action="store_true")

    parser.add_argument('--text_shift', action='store_true', help='whether to use an text shiftscaler')
    parser.add_argument('--name_path', type=str, default="./configs/cls_openearthmap.txt")
    parser.add_argument('--img_shift', action='store_true', help='whether to use an image shiftscaler')
    parser.add_argument('--do_shift', action="store_true")
    parser.add_argument('--do_scale', action="store_true")
    parser.add_argument('--do_film', action='store_true')

    parser.add_argument('--concat_concepts', action='store_true')
    parser.add_argument('--macro_pooling', action='store_true')

    parser.add_argument('--ensemble_concepts', action="store_true")
    parser.add_argument('--num_classes', type=int, default=None)
    parser.add_argument('--dataset_name', type=str, default='loveda')

    parser.add_argument('--use_susx_feats', action='store_true')

    parser.add_argument('--world_size', default=-1, type=int, help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int, help='node rank')
    parser.add_argument('--dist_url', default='env://', type=str, help='url used to set up distributed training')
    parser.add_argument('--dist_backend', default='nccl', type=str, help='distributed backend')
    parser.add_argument('--save_result', type=str, default='result.txt', help='path to save result file')

    args = parser.parse_args()

    # Prompt routing (v4):
    # - cat_prompt_source_path always preserves the original multi-description bank.
    # - name_path controls the actual segmentation-prediction prompts.
    # When Cat-Prompt/RSAP prediction is disabled, prediction falls back to one
    # category-name prompt per class, while SDR may still read the preserved
    # multi-description bank only for detached reliability estimation.
    args.cat_prompt_source_path = args.name_path
    args.reliability_prompt_path = args.cat_prompt_source_path
    if not args.module_cat_prompt:
        args.name_path = _build_single_prompt_file(args.name_path)

    # Legacy Visual Guidance and RSAP are independent alternatives. Turning off
    # legacy Visual Guidance disables text_adjust only when RSAP is also off.
    # This lets RSAP replace Visual Guidance instead of being nested inside it.
    if not args.module_visual_guidance and not args.module_rsap_v1:
        args.text_adjust = False

    if args.module_rsap_v1:
        if not args.module_cat_prompt:
            raise ValueError('--module_rsap_v1 requires Cat-Prompt / multiple prompts per class.')
        if args.text_adjust != 'True':
            raise ValueError("--module_rsap_v1 requires --text_adjust True.")
        if args.rsap_topk < 1:
            raise ValueError('--rsap_topk must be >= 1.')
        if args.rsap_gamma < 0:
            raise ValueError('--rsap_gamma must be non-negative.')

    if args.loss_sdr:
        if args.loss_src_cons:
            raise ValueError(
                '--loss_sdr already contains GSC; do not combine it with --loss_src_cons '
                'or source consistency would be counted twice.'
            )
        if args.rsap_gamma < 0:
            raise ValueError('--rsap_gamma must be non-negative.')
        if args.rsap_topk < 1:
            raise ValueError('--rsap_topk must be >= 1.')
        if not 0.0 <= args.sdr_min_weight <= 1.0:
            raise ValueError('--sdr_min_weight must be in [0, 1].')

    if args.no_reliability and not (args.module_rsap_v1 and args.loss_sdr):
        raise ValueError('--no_reliability requires Full: --module_rsap_v1 --loss_sdr.')

    return args
