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
    # parser.add_argument('--selection_p', default=0.1, type=float, help='confidence selection percentile')
    parser.add_argument('--selection_p', default=6, type=float, help='confidence selection percentile') 
    parser.add_argument('--vis_feat', default=None)  
    parser.add_argument('--tta_steps', default=1, type=int, help='test-time-adapt steps')
    parser.add_argument('--n_ctx', default=4, type=int, help='number of tunable text tokens')
    parser.add_argument('--ctx_init', default=None, type=str, help='init tunable text prompts')
    parser.add_argument('--cocoop', action='store_true', default=False, help="use cocoop's output as prompt initialization")
    parser.add_argument('--load', default=None, type=str, help='path to a pre-trained coop/cocoop')
    parser.add_argument('--seed', type=int, default=0)

    # TPS
    parser.add_argument('--img_aug', action="store_true")
    parser.add_argument('--with_concepts', action="store_true")
    parser.add_argument('--with_templates', action="store_true")
    parser.add_argument('--with_coop', action="store_true")
    parser.add_argument('--concept_type', type=str, default='gpt4', help='concepts to choose from')
    parser.add_argument('--logname', type=str)
    parser.add_argument('--loss_prompt', default=False)   
    parser.add_argument('--text_adjust', default=False)  
    parser.add_argument('--alpha', default=0.02, type = float)  
    # parser.add_argument('--alpha_cls', default=0.02, type = float)   
    parser.add_argument('--tps_entropy_scale', default=1.0, type = float)   
    
    parser.add_argument('--prob_thd', default=0.1, type = float)  
    parser.add_argument('--cls_token_lambda', default=-0.3, type = float)  
    parser.add_argument('--logit_scale', default=-0.3, type = float)  
    parser.add_argument('--logit_weight', default=-0.3, type = float) 
    parser.add_argument('--bg_idx', default=-0.3, type = float)  
    parser.add_argument('--prompt_logit_scale', default=1.0, type = float) 
    parser.add_argument('--prompt_logit_weight', default=1.0, type = float)  

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

    args = parser.parse_args()

    return args