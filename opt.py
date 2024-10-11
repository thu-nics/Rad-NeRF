import argparse

def get_opts():
    parser = argparse.ArgumentParser()

    # dataset parameters
    parser.add_argument('--root_dir', type=str, required=True,
                        help='root directory of dataset')
    parser.add_argument('--dataset_type', type=str, default='nsvf',
                        # choices=['nerf', 'nsvf', 'colmap', 'nerfpp', 'rtmv'],
                        help='which dataset type to load')
    parser.add_argument('--dataset_name', type=str, default='llff',
                        help='which dataset to train/test')
    parser.add_argument('--scene_name', type=str, default='fern',
                        help='which specified scene of the dataset to train/test')
    parser.add_argument('--split', type=str, default='train',
                        choices=['train', 'trainval', 'trainvaltest'],
                        help='use which split to train')
    parser.add_argument('--downsample', type=float, default=1.0,
                        help='downsample factor (<=1.0) for the images')

    # model parameters
    parser.add_argument('--scale', type=float, default=1,
                        help='scene scale (whole scene must lie in [-scale, scale]^3')
    parser.add_argument('--hash_table_size', type=int, default=19,
                        help='T of NGP')

    # loss parameters
    parser.add_argument('--opacity_loss_w', type=float, default=1e-3,
                        help='''weight of opacity loss (see losses.py),
                        0 to disable (default), to enable,
                        a good value is 1e-3 for real scene and ? for synthetic scene
                        ''')
    parser.add_argument('--distortion_loss_w', type=float, default=0,
                        help='''weight of distortion loss (see losses.py),
                        0 to disable (default), to enable,
                        a good value is 1e-3 for real scene and 1e-2 for synthetic scene
                        ''')
    parser.add_argument('--disp_loss_w', type=float, default=0,
                        help='''weight of disperity loss (see losses.py),
                        ''')


    # training options
    parser.add_argument('--batch_size', type=int, default=8192,
                        help='number of rays in a batch')
    parser.add_argument('--ray_sampling_strategy', type=str, default='pixel',
                        choices=['pixel', 'patch'],
                        help='''
                        pixel: uniformly from all pixels of ALL images
                        patch: uniformly from all patches of ALL image
                        ''')
    parser.add_argument('--patch_size', type=int, default=16,
                        help='size of patch image(16*16)')
    parser.add_argument('--num_epochs', type=int, default=30,
                        help='number of training epochs')
    parser.add_argument('--warmup_steps', type=int, default=256,
                        help='the iterations of warmup training')
    parser.add_argument('--num_gpus', type=int, default=1,
                        help='number of gpus')
    parser.add_argument('--num_view', type=int, default=0,
                        help='the few-shot training setting (0 means full-shot)')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='id of specified gpu')
    parser.add_argument('--lr', type=float, default=1e-2,
                        help='learning rate')

    # experimental training options
    parser.add_argument('--optimize_ext', action='store_true', default=False,
                        help='whether to optimize extrinsics')
    parser.add_argument('--random_bg', action='store_true', default=False,
                        help='''whether to train with random bg color (real scene only)
                        to avoid objects with black color to be predicted as transparent
                        ''')

    # depth priors options
    parser.add_argument("--depth_N_rand", type=int, default=4, 
                        help='batch size for depth')
    parser.add_argument("--depth_N_iters", type=int, default=201,
                        help='number of iterations for depth')
    parser.add_argument("--depth_H", type=int, default=480, 
                        help='the height of depth image (must be 16x)')
    parser.add_argument("--depth_W", type=int, default=640,
                        help='the width of depth image (must be 16x)')
    parser.add_argument("--depth_lrate", type=float, default=4e-4,
                        help='learning rate for depth')
    parser.add_argument("--depth_i_weights", type=int, default=100,
                        help='frequency of weight ckpt saving for depth')
    parser.add_argument("--depth_i_print",   type=int, default=20,
                        help='frequency of console printout and metric loggin')
    parser.add_argument('--depth_loss_w', type=float, default=0,
                        help='''weight of depth loss (see losses.py),
                        ''')
    
    # moe training options
    parser.add_argument('--moe_training', action='store_true', default=False,
                        help='whether to apply moe training')
    parser.add_argument("--model_zoo_size", type=int, default=5, 
                        help='the number of models')
    parser.add_argument('--gate_type', type=str, default='ray',
                        help='the type of gating net')
    parser.add_argument('--model_type', type=str, default='switch',
                        help='the type of model in other exps')
    parser.add_argument('--diversity_loss_w', type=float, default=0,
                        help='''weight of gate sparsity loss (see losses.py),
                        ''')
    parser.add_argument('--cv_loss_w', type=float, default=0,
                        help='''weight of gate cv loss (see losses.py),
                        ''')
    parser.add_argument('--depth_mutual_loss_w', type=float, default=0,
                        help='''weight of depth mutual loss (see losses.py),
                        ''')
    parser.add_argument('--overlap_ratio', type=float, default=0.25,
                        help='''the ratio of overlap between expert nerfs,
                        ''')
    
    # moe distillation options
    parser.add_argument('--t_ckpt_path', type=str, default=None,
                        help='pretrained teacher model checkpoint to load')
    parser.add_argument('--feat_loss_w', type=float, default=0,
                        help='''weight of feature loss (see losses.py),
                        ''')

    # validation options
    parser.add_argument('--eval_lpips', action='store_true', default=False,
                        help='evaluate lpips metric (consumes more VRAM)')
    parser.add_argument('--val_only', action='store_true', default=False,
                        help='run only validation (need to provide ckpt_path)')
    parser.add_argument('--no_save_test', action='store_true', default=False,
                        help='whether to save test image and video')

    # misc
    parser.add_argument('--exp_name', type=str, default='base',
                        help='experiment name')
    parser.add_argument('--ckpt_path', type=str, default=None,
                        help='pretrained checkpoint to load (including optimizers, etc)')
    parser.add_argument('--weight_path', type=str, default=None,
                        help='pretrained checkpoint to load (excluding optimizers, etc)')

    return parser.parse_args()
