import torch
import os
from train import define_model
from data import transform_cifar, transform_imagenet, transform_svhn, ImageFolder
from data import MultiEpochsDataLoader, save_img, TensorDataset
import torch.nn as nn
import torch.nn.functional as F
import torchvision


def remove_prefix_checkpoint(dictionary, prefix):
    keys = sorted(dictionary.keys())
    for key in keys:
        if key.startswith(prefix):
            newkey = key[len(prefix) + 1:]
            dictionary[newkey] = dictionary.pop(key)
    return dictionary


def load_ckpt(model, file_dir, verbose=True):
    checkpoint = torch.load(file_dir)
    if 'state_dict' in checkpoint:
        checkpoint = checkpoint['state_dict']
    checkpoint = remove_prefix_checkpoint(checkpoint, 'module')
    model.load_state_dict(checkpoint)

    if verbose:
        print(f"\n=> loaded checkpoint '{file_dir}'")


def load_pretrained_herding(args):
    model = define_model(args, args.nclass).cuda()
    if args.dataset == 'imagenet':
        traindir = os.path.join(args.imagenet_dir, 'train')
        valdir = os.path.join(args.imagenet_dir, 'val')
        _, test_transform = transform_imagenet(size=args.size)

        train_dataset = ImageFolder(
            traindir,
            test_transform,  # No augment here for feature extraction!
            nclass=args.nclass,
            seed=args.dseed,
            load_memory=False)
        val_dataset = ImageFolder(valdir,
                                  test_transform,
                                  nclass=args.nclass,
                                  seed=args.dseed,
                                  load_memory=False)
        if args.nclass == 100:
            file_dir = f'./results/imagenet-100/resnet10apin_cut_rrc_wd0.0001/model_best.pth.tar'
        elif args.nclass == 10:
            file_dir = f'./results/imagenet-10/resnet10apin_cut/model_best.pth.tar'
        else:
            raise AssertionError("Models not exist!")

    elif args.dataset == 'cifar10':
        _, test_transform = transform_cifar(augment=args.augment, from_tensor=False)

        # No augment here for feature extraction!
        train_dataset = torchvision.datasets.CIFAR10(args.data_dir,
                                                     train=True,
                                                     transform=test_transform)
        val_dataset = torchvision.datasets.CIFAR10(args.data_dir,
                                                   train=False,
                                                   transform=test_transform)
        file_dir = f'./results/cifar10/conv3in_cut/CIFAR10_ConvNet_Feature_dsa_cut.pt'

    elif args.dataset == 'svhn':
        _, test_transform = transform_svhn(augment=args.augment, from_tensor=False)

        # No augment here for feature extraction!
        train_dataset = torchvision.datasets.SVHN(os.path.join(args.data_dir, 'svhn'),
                                                  split='train',
                                                  transform=test_transform)
        val_dataset = torchvision.datasets.SVHN(os.path.join(args.data_dir, 'svhn'),
                                                split='test',
                                                transform=test_transform)
        if args.net_type == 'convnet':
            file_dir = f'./results/svhn/conv3in_cut/model_best.pth.tar'
        else:
            file_dir = f'./results/svhn/resnet10_cut/model_best.pth.tar'

    else:
        raise AssertionError("Dataset is not supported!")

    loader = MultiEpochsDataLoader(train_dataset,
                                   batch_size=args.batch_size // 2,
                                   shuffle=False,
                                   num_workers=args.workers,
                                   persistent_workers=args.workers > 0)
    load_ckpt(model, file_dir)

    return train_dataset, val_dataset, loader, model


def get_features(model, f_idx, loader):
    # Get features
    features = []
    targets = []
    with torch.no_grad():
        model.eval()
        for input, target in loader:
            input = input.cuda()
            target = target.cuda()

            feat = model.get_feature(input, f_idx)[0]
            feat = feat.reshape(feat.size(0), -1)

            features.append(feat)
            targets.append(target)

    features = torch.cat(features).squeeze()
    targets = torch.cat(targets)
    print("Feature shape: ", features.shape)

    return features, targets


def randomselect(dataset, ipc, nclass, targets=None):
    if targets == None:
        targets = dataset.targets
    cls_idx = [[] for _ in range(nclass)]
    for i in range(len(dataset)):
        if targets[i] < nclass:
            cls_idx[targets[i]].append(i)

    indices = []
    for c in range(nclass):
        indices += cls_idx[c][:ipc]

    return indices


def herding_select(args, features, targets, descending=False):
    # Herding
    indices_slct = []
    indices_full = torch.arange(len(features))
    for c in range(args.nclass):
        indices = targets == c
        feature_c = features[indices]
        indices_c = indices_full[indices]

        feature_mean = feature_c.mean(0, keepdim=True)
        current_sum = torch.zeros_like(feature_mean)

        cur_indices = []
        for k in range(args.ipc):
            target = (k + 1) * feature_mean - current_sum
            dist = torch.norm(target - feature_c, dim=1)
            indices_sorted = torch.argsort(dist, descending=descending)

            # We can make this faster by reducing feature matrix
            for idx in indices_sorted:
                idx = idx.item()
                if idx not in cur_indices:
                    cur_indices.append(idx)
                    break
            current_sum += feature_c[idx]

        indices_slct.append(indices_c[cur_indices])

    return indices_slct


def resol(args, img, target, max_size=-1):
    resize = nn.Upsample(size=args.size, mode='bilinear')
    data = resize(F.interpolate(img, size=(args.size // args.factor, args.size // args.factor)))

    return data, target


def herding(args):
    train_dataset, val_dataset, loader, model = load_pretrained_herding(args)
    if args.dataset == 'imagenet':
        f_idx = 5
    else:
        f_idx = 2

    features, targets = get_features(model, f_idx, loader)
    indices_slct = herding_select(args, features, targets)

    # Select and make dataset
    data = []
    target = []
    indices_slct = torch.cat(indices_slct)
    for i in indices_slct:
        img, lab = train_dataset[i]
        data.append(img)
        target.append(lab)
    data = torch.stack(data)
    target = torch.tensor(target)
    print("Herding data selected! ", data.shape)

    if args.dataset == 'imagenet':
        if args.factor > 1:
            data, target = resol(args, data, target)
            print("Resolution reduced!", data.shape)
        train_transform, _ = transform_imagenet(augment=args.augment,
                                                from_tensor=True,
                                                normalize=False,
                                                size=0,
                                                rrc=args.rrc,
                                                rrc_size=args.size)
    elif args.dataset == 'svhn':
        train_transform, _ = transform_svhn(augment=args.augment, normalize=False, from_tensor=True)
    elif args.dataset[:5] == 'cifar':
        train_transform, _ = transform_cifar(augment=args.augment,
                                             normalize=False,
                                             from_tensor=True)
    else:
        train_transform = None

    train_dataset = TensorDataset(data, target, train_transform)

    save_img('./results/herding.png',
             torch.stack([d[0] for d in train_dataset]),
             dataname=args.dataset)
    return train_dataset, val_dataset


if __name__ == '__main__':
    import torch.nn as nn
    from argument import args
    from test import validate

    train_dataset, val_dataset, loader, model = load_pretrained_herding(args)
    val_loader = MultiEpochsDataLoader(val_dataset,
                                       batch_size=args.batch_size // 2,
                                       shuffle=False,
                                       persistent_workers=True,
                                       num_workers=4)
    criterion = nn.CrossEntropyLoss()

    top1, top5, _ = validate(args, val_loader, model, criterion, 0, logger=print)
    print(top1)