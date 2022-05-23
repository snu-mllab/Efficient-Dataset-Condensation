import argparse
import os
import sys


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="cifar100", type=str)
parser.add_argument(
    "--data_dir",
    default="/data_large/readonly",
    type=str,
    help="directory that containing dataset, except imagenet (see data.py)",
)
parser.add_argument("--num_classes", default=100, type=int)
parser.add_argument(
    "--nb_cl_fg", default=20, type=int, help="the number of classes in first group"
)
parser.add_argument("--nb_cl", default=20, type=int, help="Classes per group")
parser.add_argument(
    "--nb_runs",
    default=1,
    type=int,
    help="Number of runs (random ordering of classes at each run)",
)
parser.add_argument(
    "--ckp_prefix",
    default="cifar100",
    type=str,
    help="Checkpoint prefix",
)
parser.add_argument("--epochs", default=1000, type=int, help="Epochs")
parser.add_argument("--T", default=2, type=float, help="Temperature for distialltion")
parser.add_argument("--beta", default=1.0, type=float, help="Beta for distialltion")
parser.add_argument("--train_batch_size", default=64, type=int, help="Train batch size")
parser.add_argument("--test_batch_size", default=100, type=int, help="Test batch size")
parser.add_argument("--eval_batch_size", default=128, type=int, help="Eval batch size")
parser.add_argument("--base_lr", default=0.01, type=float, help="Initial learning rate")
parser.add_argument(
    "--lr_strat",
    default=[600, 800],
    nargs="*",
    type=int,
    help="Epochs where learning rate gets decreased",
)
parser.add_argument(
    "--lr_factor", default=0.2, type=float, help="Learning rate decrease factor"
)
parser.add_argument(
    "--custom_weight_decay", default=5e-4, type=float, help="Weight Decay"
)
parser.add_argument("--custom_momentum", default=0.9, type=float, help="Momentum")
parser.add_argument(
    "--exemplar",
    default="condense",
    choices=["condense", "dm", "dsa", "herding"],
)
parser.add_argument("--ipc", default=20, choices=[20], type=int)
parser.add_argument("--factor", default=1, choices=[1, 2], type=int)
parser.add_argument("--strong_aug", default=True, type=str2bool)
parser.add_argument("--mix_p", default=0.5, type=float)
parser.add_argument(
    "--net",
    default="convnet",
    choices=["convnet", "resnet10"],
    type=str,
)


args = parser.parse_args()

# Assertions
assert len(args.lr_strat) == 2
assert args.nb_cl_fg == 20
assert args.nb_cl == 20

if args.exemplar.startswith("condense"):
    args.nb_protos = args.ipc * (args.factor**2)
else:
    args.nb_protos = args.ipc


# For DSA, below are the settings with best performance.
# - We use bigger batch_size
# - And deactivate CutMix with `mix_p = 0.0`
if args.exemplar == "dsa":
    assert args.train_batch_size == 256
    assert args.mix_p == 0.0
    assert args.strong_aug == True

args.ckp_prefix = (
    f"{args.ckp_prefix}_net_{args.net}_exemplar_{args.exemplar}_ipc_{args.ipc}"
)

if args.factor != 1:
    args.ckp_prefix += f"_factor_{args.factor}"
