import argparse


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
    vgta_group = parser.add_mutually_exclusive_group()
    vgta_group.add_argument(
        '--module_vgta',
        dest='module_vgta',
        action='store_true',
        help='Enable TMPA Visual-Guided Test-Time Adaptation (default).',
    )
    vgta_group.add_argument(
        '--no_module_vgta',
        dest='module_vgta',
        action='store_false',
        help=(
            'Disable VGTA. Visual-guided text adjustment and learnable text-shift '
            'adaptation are bypassed even if their legacy flags are present.'
        ),
    )
    parser.set_defaults(module_cat_prompt=True, module_vgta=True)

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
    parser.add_argument('--rank', default=-1, type=int, help='node rank for distributed training')
    parser.add_argument('--dist_url', default='env://', type=str, help='url used to set up distributed training')
    parser.add_argument('--dist_backend', default='nccl', type=str, help='distributed backend')
    parser.add_argument('--save_result', type=str, default='result.txt', help='path to save result file')

    return parser.parse_args()
