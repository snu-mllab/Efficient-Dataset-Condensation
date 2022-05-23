import torch
import torchvision
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
import argparse
import pickle


def compute_accuracy(tg_model, evalloader, scale=None, device=None):
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tg_model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for _, (inputs, targets) in enumerate(evalloader):
            inputs, targets = inputs.to(device), targets.to(device)
            total += targets.size(0)

            outputs = tg_model(inputs)
            outputs = F.softmax(outputs, dim=1)
            if scale is not None:
                assert scale.shape[0] == 1
                assert outputs.shape[1] == scale.shape[1]
                outputs = outputs / scale.repeat(outputs.shape[0], 1).type(
                    torch.FloatTensor
                ).to(device)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()

    cnn_acc = 100.0 * correct / total

    return [cnn_acc]


if __name__ == "__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckp_prefix",
        default="checkpoints/cifar100_net_convnet_exemplar_dm_ipc_20_run_0_",
        type=str,
    )
    parser.add_argument(
        "--data_dir",
        default="/data_large/readonly",
        type=str,
        help="directory that containing dataset, except imagenet (see data.py)",
    )
    parser.add_argument("--order", default="checkpoints/cifar100_order.pkl", type=str)
    parser.add_argument(
        "--nb_cl_fg", default=20, type=int, help="the number of classes in first group"
    )
    parser.add_argument("--nb_cl", default=20, type=int, help="Classes per group")
    args = parser.parse_args()

    print("###########################################################")
    print()
    print(args.ckp_prefix)

    with open(args.order, "rb") as f:
        order = pickle.load(f)

    order_list = list(order)
    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4866, 0.4409), (0.2009, 0.1984, 0.2023)),
        ]
    )
    evalset = torchvision.datasets.CIFAR100(
        root=args.data_dir,
        train=False,
        download=False,
        transform=transform_test,
    )
    input_data = evalset.data
    input_labels = evalset.targets
    map_input_labels = np.array([order_list.index(i) for i in input_labels])

    cnn_cumul_acc = []
    num_classes = []
    nb_cl = args.nb_cl
    start_iter = int(args.nb_cl_fg / nb_cl) - 1

    for iteration in range(start_iter, int(100 / nb_cl)):
        ckp_name = f"{args.ckp_prefix}iteration_{iteration}_model.pt"
        tg_model = torch.load(ckp_name)
        indices = np.array(
            [i in range(0, (iteration + 1) * nb_cl) for i in map_input_labels]
        )

        evalset.data = input_data[indices]
        evalset.targets = map_input_labels[indices]
        evalloader = torch.utils.data.DataLoader(
            evalset, batch_size=128, shuffle=False, num_workers=2
        )
        acc = compute_accuracy(tg_model, evalloader)
        cnn_cumul_acc.append(acc[0])
        num_classes.append((iteration + 1) * nb_cl)

    print("###########################################################")
    print()
    print(" Acc in each stage [avg acc]")
    print()
    print("###########################################################")
    print()
    print(" ", end="")
    for i in range(len(cnn_cumul_acc)):
        print("{:.2f}".format(cnn_cumul_acc[i]), end="")
        if i != len(cnn_cumul_acc) - 1:
            print(" / ", end="")

    print("  [{:.2f}] ".format(np.mean(cnn_cumul_acc)))
    print()
    print("###########################################################")
    print()
    print(" Number of classes", num_classes)
    print()
    print("###########################################################")
