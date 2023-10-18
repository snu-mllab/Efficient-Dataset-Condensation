"""
The original code is created by Jang-Hyun Kim.
GitHub Repository: https://github.com/snu-mllab/Efficient-Dataset-Condensation
"""
import argparse
from misc.reproduce import set_arguments


def str2bool(v):
    """Cast string to boolean
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def ipc_epoch(ipc, factor, nclass=10, bound=-1):
    """Calculating training epochs for ImageNet
    """
    factor = max(factor, 1)
    ipc *= factor**2
    if bound > 0:
        ipc = min(ipc, bound)

    if ipc == 1:
        epoch = 3000
    elif ipc <= 10:
        epoch = 2000
    elif ipc <= 50:
        epoch = 1500
    elif ipc <= 200:
        epoch = 1000
    elif ipc <= 500:
        epoch = 500
    else:
        epoch = 300

    if nclass == 100:
        epoch = int((2 / 3) * epoch)
        epoch = epoch - (epoch % 100)

    return epoch


def tune_lr_img(args, lr_img):
    """Tuning lr_img for imagenet 
    """
    # Use mse loss for 32x32 img and ConvNet
    ipc_base = 10
    if args.dataset == 'imagenet':
        imsize_base = 224
    elif args.dataset == 'speech':
        imsize_base = 64
    elif args.dataset == 'mnist':
        imsize_base = 28
    else:
        imsize_base = 32

    param_ratio = (args.ipc / ipc_base)
    if args.size > 0:
        param_ratio *= (args.size / imsize_base)**2

    lr_img = lr_img * param_ratio
    return lr_img


def remove_aug(augtype, remove_aug):
    """Remove certain type of augmentation (string)
    """
    aug_list = []
    for aug in augtype.split("_"):
        if aug not in remove_aug.split("_"):
            aug_list.append(aug)

    return "_".join(aug_list)


parser = argparse.ArgumentParser(description='')
# Dataset
parser.add_argument('-d',
                    '--dataset',
                    default='cifar10',
                    type=str,
                    help='dataset (options: mnist, fashion, svhn, cifar10, cifar100, and imagenet)')
parser.add_argument('--data_dir',
                    default='/data_large/readonly',
                    type=str,
                    help='directory that containing dataset, except imagenet (see data.py)')
parser.add_argument('--imagenet_dir', default='/ssd_data/imagenet/', type=str)
parser.add_argument('--nclass', default=10, type=int, help='number of classes in trianing dataset')
parser.add_argument('--dseed', default=0, type=int, help='seed for class sampling')
parser.add_argument('--size', default=224, type=int, help='spatial size of image')
parser.add_argument('--phase', default=-1, type=int, help='index for multi-processing')
parser.add_argument('--nclass_sub', default=-1, type=int, help='number of classes for each process')
parser.add_argument('-l',
                    '--load_memory',
                    type=str2bool,
                    default=True,
                    help='load training images on the memory')
# Network
parser.add_argument('-n',
                    '--net_type',
                    default='convnet',
                    type=str,
                    help='network type: resnet, resnet_ap, convnet')
parser.add_argument('--norm_type',
                    default='instance',
                    type=str,
                    choices=['batch', 'instance', 'sn', 'none'])
parser.add_argument('--depth', default=10, type=int, help='depth of the network')
parser.add_argument('--width', default=1.0, type=float, help='width of the network')

# Training
parser.add_argument('--epochs', default=300, type=int, help='number of training epochs')
parser.add_argument('--batch_size', default=64, type=int, help='mini-batch size for training')
parser.add_argument('--lr', default=0.01, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float, help='weight decay')
parser.add_argument('--seed', default=0, type=int, help='random seed for training')
parser.add_argument('--pretrained', action='store_true')

# Mixup
parser.add_argument('--mixup',
                    default='cut',
                    type=str,
                    choices=('vanilla', 'cut'),
                    help='mixup choice for evaluation')
parser.add_argument('--mixup_net',
                    default='cut',
                    type=str,
                    choices=('vanilla', 'cut'),
                    help='mixup choice for training networks in condensation stage')
parser.add_argument('--beta', default=1.0, type=float, help='mixup beta distribution')
parser.add_argument('--mix_p', default=1.0, type=float, help='mixup probability')

# Logging
parser.add_argument('--print-freq',
                    '-p',
                    default=10,
                    type=int,
                    help='print frequency (default: 10)')
parser.add_argument('--verbose',
                    dest='verbose',
                    action='store_true',
                    help='to print the status at every iteration')
parser.add_argument('-j', '--workers', default=8, type=int, help='number of data loading workers')
parser.add_argument('--save_ckpt', type=str2bool, default=False)
parser.add_argument('--tag', default='', type=str, help='name of experiment')
parser.add_argument('--test', action='store_true', help='for debugging, do not save results')
parser.add_argument('--time', action='store_true', help='measuring time for each step')

# Condense
parser.add_argument('-i', '--ipc', type=int, default=-1, help='number of condensed data per class')
parser.add_argument('-f',
                    '--factor',
                    type=int,
                    default=1,
                    help='multi-formation factor. (1 for IDC-I)')
parser.add_argument('--decode_type',
                    type=str,
                    default='single',
                    choices=['single', 'multi', 'bound'],
                    help='multi-formation type')
parser.add_argument('--init',
                    type=str,
                    default='random',
                    choices=['random', 'noise', 'mix'],
                    help='condensed data initialization type')
parser.add_argument('-a',
                    '--aug_type',
                    type=str,
                    default='color_crop_cutout',
                    help='augmentation strategy for condensation matching objective')
## Matching objective
parser.add_argument('--match',
                    type=str,
                    default='grad',
                    choices=['feat', 'grad'],
                    help='feature or gradient matching')
parser.add_argument('--metric',
                    type=str,
                    default='l1',
                    choices=['mse', 'l1', 'l1_mean', 'l2', 'cos'],
                    help='matching objective')
parser.add_argument('--bias', type=str2bool, default=False, help='match bias or not')
parser.add_argument('--fc', type=str2bool, default=False, help='match fc layer or not')
parser.add_argument('--f_idx',
                    type=str,
                    default='4',
                    help='feature matching layer. comma separation')
## Optimization
# For small datasets, niter=2000 is enough for the full convergence.
# For faster optimzation, you can early stop the code based on the printed log.
parser.add_argument('--niter', type=int, default=500, help='number of outer iteration')
parser.add_argument('--inner_loop', type=int, default=100, help='number of inner iteration')
parser.add_argument('--early',
                    type=int,
                    default=0,
                    help='number of pretraining epochs for condensation networks')
parser.add_argument('--fix_iter',
                    type=int,
                    default=-1,
                    help='number of outer iteration maintaining the condensation networks')
parser.add_argument('--net_epoch',
                    type=int,
                    default=1,
                    help='number of epochs for training network at each inner loop')
parser.add_argument('--n_data',
                    type=int,
                    default=500,
                    help='number of samples for training network at each inner loop')
parser.add_argument('--pt_from', type=int, default=-1, help='pretrained networks index')
parser.add_argument('--pt_num', type=int, default=1, help='pretrained networks range')
parser.add_argument('--batch_real',
                    type=int,
                    default=64,
                    help='batch size of real training data used for matching')
parser.add_argument(
    '--batch_syn_max',
    type=int,
    default=128,
    help=
    'maximum number of synthetic data used for each matching (ramdom sampling for large synthetic data)'
)
parser.add_argument('--lr_img', type=float, default=5e-3, help='condensed data learning rate')
parser.add_argument('--mom_img', type=float, default=0.5, help='condensed data momentum')
parser.add_argument('--reproduce', action='store_true', help='for reproduce our setting')

# Test
parser.add_argument('-s',
                    '--slct_type',
                    type=str,
                    default='idc',
                    help='data condensation type (idc, dsa, kip, random, herding)')
parser.add_argument('--repeat', default=1, type=int, help='number of test repetetion')
parser.add_argument('--dsa',
                    type=str2bool,
                    default=False,
                    help='Use DSA augmentation for evaluation or not')
parser.add_argument('--dsa_strategy', type=str, default='color_crop_cutout_flip_scale_rotate')
parser.add_argument('--rrc',
                    type=str2bool,
                    default=True,
                    help='use random resize crop for ImageNet')
parser.add_argument('--same_compute',
                    type=str2bool,
                    default=False,
                    help='match evaluation training steps for IDC')
parser.add_argument('--name', type=str, default='', help='name of the test data folder')

parser.set_defaults(bottleneck=True)
parser.set_defaults(verbose=False)
args = parser.parse_args()

if args.reproduce:
    args = set_arguments(args)
""" 
DATA 
"""
args.nch = 3
if args.dataset[:5] == 'cifar':
    args.size = 32
    args.mix_p = 0.5
    args.dsa = True
    if args.dataset == 'cifar10':
        args.nclass = 10
    elif args.dataset == 'cifar100':
        args.nclass = 100

if args.dataset == 'svhn':
    args.size = 32
    args.nclass = 10
    args.mix_p = 0.5
    args.dsa = True
    args.dsa_strategy = remove_aug(args.dsa_strategy, 'flip')

if args.dataset[:5] == 'mnist':
    args.nclass = 10
    args.size = 28
    args.nch = 1
    args.mix_p = 0.5
    args.dsa = True
    args.dsa_strategy = remove_aug(args.dsa_strategy, 'flip')

if args.dataset == 'fashion':
    args.nclass = 10
    args.size = 28
    args.nch = 1
    args.mix_p = 0.5
    args.dsa = True

if args.dataset == 'imagenet':
    if args.net_type == 'convnet':
        args.net_type = 'resnet_ap'
    args.size = 224
    if args.nclass >= 100:
        args.load_memory = False
        print("args.load_memory is setted as False! (see args.argument)")
        # We need to tune lr and weight decay
        args.lr = 0.1
        args.weight_decay = 1e-4
        args.batch_size = max(128, args.batch_size)
        args.batch_real = max(128, args.batch_real)

if args.dataset == 'speech':
    args.nch = 1
    args.size = 64
    if args.net_type == 'convnet':
        args.depth = 4
    args.nclass = 8
    # For speech data, I didn't use data augmentation
    args.mixup = 'vanilla'
    args.mixup_net = 'vanilla'
    args.dsa = False

datatag = f'{args.dataset}'
if args.dataset == 'imagenet':
    datatag += f'{args.nclass}'
    if args.dseed != 0:
        datatag += f'-seed{args.dseed}'
"""
Network
"""
if args.net_type == 'convnet':
    if args.depth > 4:
        args.depth = 3
    args.f_idx = str(args.depth - 1)

modeltag = f'{args.net_type}{args.depth}'
if args.net_type == 'resnet_ap':
    modeltag = f'resnet{args.depth}ap'
if args.net_type == 'convnet':
    modeltag = f'conv{args.depth}'
if args.norm_type == 'instance':
    modeltag += 'in'
if args.width != 1.0:
    modeltag += f'_w{args.width}'
"""
EXP tag (folder name)
"""
# Default initialization for multi-formation
if args.factor > 1:
    args.init = 'mix'

if args.tag != '':
    args.tag = f'_{args.tag}'
if args.ipc > 0:
    if args.slct_type == 'random':
        args.tag += f'_rand{args.ipc}'

    elif args.slct_type == 'idc':
        # Matching
        if args.match == 'feat':
            args.tag += f'_f{args.f_idx}'
            f_list = [int(s) for s in args.f_idx.split(',')]
            if len(f_list) == 1:
                f_list.append(-1)
            args.idx_from, args.idx_to = f_list
            args.metric = 'mse'
        else:
            args.tag += f'_{args.match}'
            if args.bias:
                args.tag += '_b'
            if args.fc:
                args.tag += '_fc'

        # Net update
        args.tag += f'_{args.metric}'
        if args.pt_from >= 0:
            args.tag += f'_pt{args.pt_from}'
            if args.pt_num > 1:
                args.tag += f'_{args.pt_num}'
        if args.fix_iter > 0:
            args.tag += f'_fix{args.fix_iter}'
        if args.early > 0:
            args.tag += f'_ely{args.early}'
        if args.n_data >= 0:
            args.tag += f'_nd{args.n_data}'
            if args.inner_loop != 100:
                args.tag += f'_inloop{args.inner_loop}'
        if args.mixup_net == 'cut':
            args.tag += f'_cut'
        if args.lr != 0.01:
            args.tag += f'_nlr{args.lr}'
        if args.weight_decay != 5e-4:
            args.tag += f'_wd{args.weight_decay}'
        if args.niter != 500:
            args.tag += f'_niter{args.niter}'

        # Multi-formation & Augmentation
        if args.factor > 0:
            args.tag += f'_factor{args.factor}'
            if args.decode_type != 'single':
                args.tag += f'_{args.decode_type}'
        if args.aug_type != 'color_crop_cutout':
            args.tag += f'_{args.aug_type}'

        # Img update
        args.tag += f'_lr{args.lr_img}'
        args.lr_img = tune_lr_img(args, args.lr_img)
        print(f"lr_img tuned! {args.lr_img:.5f}")
        if args.momentum != 0.9:
            args.tag += f'_mom{args.momentum}'
        if args.batch_real != 64:
            args.tag += f'_b_real{args.batch_real}'
        if args.batch_syn_max != 128:
            args.tag += f'_synmax{args.batch_syn_max}'

        args.tag += f'_{args.init}'
        args.tag += f'_ipc{args.ipc}'

        # For multi-processing (class partitioning)
        if args.nclass_sub > 0:
            args.tag += f'_{args.nclass_sub}'
        if args.phase >= 0:
            args.tag += f'_phase{args.phase}'
else:
    if args.mixup != 'vanilla':
        args.tag += f'_{args.mixup}'

# Result folder name
if args.test:
    args.save_dir = './results/test'
else:
    args.save_dir = f"./results/{datatag}/{modeltag}{args.tag}"
args.modeltag = modeltag
args.datatag = datatag
"""
Evaluation setting
"""
# Setting evaluation training epochs
if args.ipc > 0:
    if args.dataset == 'imagenet':
        if args.decode_type == 'bound':
            args.epochs = ipc_epoch(args.ipc, args.factor, args.nclass, bound=args.batch_syn_max)
        else:
            args.epochs = ipc_epoch(args.ipc, args.factor, args.nclass)
        args.epoch_print_freq = args.epochs // 100
    else:
        args.epochs = 1000
        args.epoch_print_freq = args.epochs
else:
    args.epoch_print_freq = 1

# Setting augmentation
if args.mixup == 'cut':
    args.dsa_strategy = remove_aug(args.dsa_strategy, 'cutout')
if args.dsa:
    args.augment = False
    print("DSA strategy: ", args.dsa_strategy)
else:
    args.augment = True
