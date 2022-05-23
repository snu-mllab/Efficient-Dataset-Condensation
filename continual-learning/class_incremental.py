import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import transforms
import numpy as np
import os
import copy
import pickle
import kornia

from models import resnet, convnet
from exemplar import Baseline, Condense
from torchvision.utils import save_image
from utils.incremental_train_and_eval import incremental_train_and_eval
from dataset import TensorCIFAR100

from argument import args


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_data():
    trainset = TensorCIFAR100(args.data_dir, train=True, download=True, transform=None)
    testset = TensorCIFAR100(args.data_dir, train=False, download=True, transform=None)
    evalset = TensorCIFAR100(args.data_dir, train=False, download=False, transform=None)

    return trainset, evalset, testset


def initialize_model(iteration, start_iter, tg_model, args):

    # init model
    if iteration == start_iter:
        last_iter = 0
        ref_model = None
    else:
        last_iter = iteration
        ref_model = copy.deepcopy(tg_model)
        in_features = tg_model.fc.in_features

        if args.net in ["resnet10", "convnet"]:
            out_features = tg_model.fc.out_features
            new_fc = nn.Linear(in_features, out_features + args.nb_cl)
            new_fc.weight.data[:out_features] = tg_model.fc.weight.data
            new_fc.bias.data[:out_features] = tg_model.fc.bias.data
        else:
            raise NotImplementedError()
        tg_model.fc = new_fc

    return last_iter, ref_model


def load_training_data(
    order_list, X_train, Y_train, X_valid_cumul, Y_valid_cumul, trainset, testset
):

    if args.strong_aug:
        transform_train = transforms.Compose(
            [
                transforms.Normalize(
                    (0.5071, 0.4866, 0.4409), (0.2009, 0.1984, 0.2023)
                ),
            ]
        )
    else:
        transform_train = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.Normalize(
                    (0.5071, 0.4866, 0.4409), (0.2009, 0.1984, 0.2023)
                ),
            ]
        )
    transform_test = transforms.Compose(
        [
            transforms.Normalize((0.5071, 0.4866, 0.4409), (0.2009, 0.1984, 0.2023)),
        ]
    )

    print(f"Batch of classes number {iteration + 1} arrives ...")
    map_Y_train = np.array([order_list.index(i) for i in Y_train])
    map_Y_valid_cumul = np.array([order_list.index(i) for i in Y_valid_cumul])

    trainset.data = transform_train(X_train).to(device)
    trainset.targets = torch.tensor(map_Y_train).to(device)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.train_batch_size, shuffle=True, num_workers=0
    )
    testset.data = transform_test(X_valid_cumul).to(device)
    testset.targets = torch.tensor(map_Y_valid_cumul).to(device)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=args.test_batch_size, shuffle=False, num_workers=0
    )
    print(f"Max and Min of train labels: {min(map_Y_train)}, {max(map_Y_train)}")
    print(
        f"Max and Min of valid labels: {min(map_Y_valid_cumul)}, {max(map_Y_valid_cumul)}"
    )

    return trainloader, testloader


def train(
    tg_model, iteration_total, iteration, start_iter, ref_model, trainloader, testloader
):
    ckp_name = f"./checkpoints/{args.ckp_prefix}_run_{iteration_total}_iteration_{iteration}_model.pt"
    print("ckp_name", ckp_name)

    tg_params = tg_model.parameters()

    tg_model = tg_model.to(device)
    if iteration > start_iter:
        ref_model = ref_model.to(device)
    tg_optimizer = optim.SGD(
        tg_params,
        lr=args.base_lr,
        momentum=args.custom_momentum,
        weight_decay=args.custom_weight_decay,
    )
    tg_lr_scheduler = lr_scheduler.MultiStepLR(
        tg_optimizer, milestones=args.lr_strat, gamma=args.lr_factor
    )

    tg_model = incremental_train_and_eval(
        args.epochs,
        tg_model,
        ref_model,
        tg_optimizer,
        tg_lr_scheduler,
        trainloader,
        testloader,
        iteration,
        start_iter,
        args.T,
        args.beta,
        args.strong_aug,
        args.mix_p,
        args.exemplar == "dsa",
    )

    if not os.path.isdir("checkpoints"):
        os.mkdir("checkpoints")
    torch.save(tg_model, ckp_name)


def save_img(save_dir, img, max_num=200, size=64, nrow=10):

    img = img[:max_num].detach()
    img = torch.clamp(img, min=0.0, max=1.0)

    if img.shape[-1] > size:
        img = kornia.geometry.transform.resize(img, size)
    save_image(img.cpu(), save_dir, nrow=nrow)


def save_data(X_protoset_cumul, Y_protoset_cumul, args):

    # # Code for visualizing prototypes
    #
    # dirname = os.path.join("visual", os.path.dirname(args.ckp_prefix))
    # os.makedirs(dirname, exist_ok=True)
    #
    # if iteration == 0:
    #     save_img(
    #         f"./visual/{args.ckp_prefix}_iteration_{iteration}_images.png",
    #         X_protoset_cumul,
    #         nrow=args.nb_protos,
    #     )

    dirname = os.path.join("checkpoints", os.path.dirname(args.ckp_prefix))
    os.makedirs(dirname, exist_ok=True)

    torch.save(
        (X_protoset_cumul, Y_protoset_cumul),
        f"./checkpoints/{args.ckp_prefix}_run_{iteration_total}_iteration_{iteration}_protoset_tensor.pt",
    )


def phase(
    order,
    iteration,
    start_iter,
    tg_model,
    trainset,
    testset,
    X_valid_total,
    Y_valid_total,
    valid_data,
    exemplar,
):

    # Exemplars
    last_iter, ref_model = initialize_model(iteration, start_iter, tg_model, args)

    print(f"{args.exemplar} : updating exemplar set...")
    # Prepare the protoset
    X_protoset_cumul = torch.tensor([])
    Y_protoset_cumul = torch.tensor([])

    exemplar.compute_prototypes(
        start_iter=start_iter,
        iteration=iteration,
        nb_cl_fg=args.nb_cl_fg,
        nb_cl=args.nb_cl,
        order=list(order),
    )

    for iteration2 in range(iteration + 1):
        current_cl = order[
            range(iteration2 * args.nb_cl, (iteration2 + 1) * args.nb_cl)
        ]
        for iter_dico in range(args.nb_cl):
            class_index = iteration2 * args.nb_cl + iter_dico

            X_protoset_cumul, Y_protoset_cumul = exemplar.concat_prototypes(
                class_index,
                current_cl,
                args.nb_cl,
                X_protoset_cumul,
                Y_protoset_cumul,
            )

    X_valid_cumul, Y_valid_cumul, X_valid_ori, Y_valid_ori = valid_data
    indices_test = np.array(
        [
            i.item()
            in order[range(last_iter * args.nb_cl, (iteration + 1) * args.nb_cl)]
            for i in Y_valid_total
        ]
    )

    X_valid = X_valid_total[indices_test]
    X_valid_cumul = torch.cat([X_valid_cumul, X_valid])

    Y_valid = Y_valid_total[indices_test]
    Y_valid_cumul = torch.cat([Y_valid_cumul, Y_valid])

    # Add the stored exemplars to the training data
    if iteration == start_iter:
        X_valid_ori = X_valid
        Y_valid_ori = Y_valid

    X_train = X_protoset_cumul
    Y_train = Y_protoset_cumul

    trainloader, testloader = load_training_data(
        list(order), X_train, Y_train, X_valid_cumul, Y_valid_cumul, trainset, testset
    )

    train(
        tg_model,
        iteration_total,
        iteration,
        start_iter,
        ref_model,
        trainloader,
        testloader,
    )

    exemplar.set_feature_model(tg_model=tg_model)

    save_data(X_protoset_cumul, Y_protoset_cumul, args)

    return X_valid_cumul, Y_valid_cumul, X_valid_ori, Y_valid_ori


if __name__ == "__main__":

    trainset, evalset, testset = load_data()

    # Initialization
    dictionary_size = args.nb_protos

    X_train_total = trainset.data
    Y_train_total = trainset.targets
    X_valid_total = testset.data
    Y_valid_total = testset.targets

    # Launch the different runs
    for iteration_total in range(args.nb_runs):
        # Select the order for the class learning
        order_name = f"./checkpoints/{args.dataset}_order.pkl"
        print(f"Order name:{order_name}")
        if os.path.exists(order_name):
            print("Loading orders")
            with open(order_name, "rb") as f:
                order = pickle.load(f)
        else:
            print("The order doesn't exist")

        print(list(order))

        # Initialization of the variables for this run
        X_valid_cumul = torch.tensor([])
        Y_valid_cumul = torch.tensor([])
        X_protoset_cumul = torch.tensor([])
        Y_protoset_cumul = torch.tensor([])
        X_valid_ori = torch.tensor([])
        Y_valid_ori = torch.tensor([])

        prototypes = torch.zeros(
            (
                args.num_classes,
                dictionary_size,
                X_train_total.shape[1],
                X_train_total.shape[2],
                X_train_total.shape[3],
            )
        )

        start_iter = int(args.nb_cl_fg / args.nb_cl) - 1
        end_iter = int(args.num_classes / args.nb_cl)

        if args.exemplar == "condense":
            exemplar = Condense(
                prototypes=prototypes,
                evalset=evalset,
                ipc=args.ipc,
                factor=args.factor,
                strong_aug=args.strong_aug,
            )
        elif args.exemplar in ["dm", "dsa", "herding"]:
            exemplar = Baseline(
                prototypes=prototypes, evalset=evalset, type=args.exemplar, net=args.net
            )
        else:
            raise NotImplementedError()

        if args.net == "resnet10":
            tg_model = resnet.ResNet("cifar100", 10, num_classes=args.nb_cl_fg, size=32)
        elif args.net == "convnet":
            tg_model = convnet.ConvNet(num_classes=args.nb_cl_fg)
        else:
            raise NotImplementedError()

        for iteration in range(start_iter, end_iter):
            valid_data = X_valid_cumul, Y_valid_cumul, X_valid_ori, Y_valid_ori
            X_valid_cumul, Y_valid_cumul, X_valid_ori, Y_valid_ori = phase(
                order,
                iteration,
                start_iter,
                tg_model,
                trainset,
                testset,
                X_valid_total,
                Y_valid_total,
                valid_data,
                exemplar,
            )
