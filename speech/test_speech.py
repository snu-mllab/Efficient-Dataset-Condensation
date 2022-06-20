import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset
from train import define_model, train
from data import TensorDataset, MultiEpochsDataLoader
from speech import load_logspec, save_log_spec
import models.resnet as RN
from math import ceil
from coreset import randomselect


def resnet10_in(args, nclass, logger=None):
    model = RN.ResNet(args.dataset, 10, nclass, 'instance', args.size, n_ch=args.n_ch)
    if logger is not None:
        logger(f"=> creating model resnet-10, norm: instance")
    return model


def resnet10_bn(args, nclass, logger=None):
    model = RN.ResNet(args.dataset, 10, nclass, 'batch', args.size, n_ch=args.n_ch)
    if logger is not None:
        logger(f"=> creating model resnet-10, norm: batch")
    return model


def load_ckpt(model, file_dir, verbose=True):
    checkpoint = torch.load(file_dir)
    if 'state_dict' in checkpoint:
        checkpoint = checkpoint['state_dict']
    checkpoint = remove_prefix_checkpoint(checkpoint, 'module')
    model.load_state_dict(checkpoint)

    if verbose:
        print(f"\n=> loaded checkpoint '{file_dir}'")


def remove_prefix_checkpoint(dictionary, prefix):
    keys = sorted(dictionary.keys())
    for key in keys:
        if key.startswith(prefix):
            newkey = key[len(prefix) + 1:]
            dictionary[newkey] = dictionary.pop(key)
    return dictionary


def decode_zoom(img, target, factor, size=-1):
    if size == -1:
        size = img.shape[-1]
    resize = nn.Upsample(size=size, mode='bilinear')

    w = img.shape[-1]
    remained = w % factor
    if remained > 0:
        img = F.pad(img, pad=(0, factor - remained), value=0.)
    s_crop = ceil(w / factor)
    n_crop = factor

    cropped = []
    for i in range(factor):
        w_loc = i * s_crop
        cropped.append(img[:, :, :, w_loc:w_loc + s_crop])
    cropped = torch.cat(cropped)
    data_dec = resize(cropped)
    target_dec = torch.cat([target for _ in range(n_crop)])

    return data_dec, target_dec


def decode_fn(data, target, factor, decode_type, bound=128):
    if factor > 1:
        data, target = decode_zoom(data, target, factor)

    return data, target


def decode(args, data, target):
    data_dec = []
    target_dec = []
    ipc = len(data) // args.nclass
    for c in range(args.nclass):
        idx_from = ipc * c
        idx_to = ipc * (c + 1)
        data_ = data[idx_from:idx_to].detach()
        target_ = target[idx_from:idx_to].detach()
        data_, target_ = decode_fn(data_,
                                   target_,
                                   args.factor,
                                   args.decode_type,
                                   bound=args.batch_syn_max)
        data_dec.append(data_)
        target_dec.append(target_)

    data_dec = torch.cat(data_dec)
    target_dec = torch.cat(target_dec)

    print("Dataset is decoded! ", data_dec.shape)
    return data_dec, target_dec


def load_pretrained_herding(args):
    # Herding pretrained model
    model = define_model(args, args.nclass).cuda()
    train_dataset, val_dataset, mean, std = load_logspec(args)
    file_dir = f'./results/speech/conv4in/model_best.pth.tar'

    loader = MultiEpochsDataLoader(train_dataset,
                                   batch_size=args.batch_size // 2,
                                   shuffle=False,
                                   num_workers=args.workers,
                                   persistent_workers=args.workers > 0)
    load_ckpt(model, file_dir)

    return train_dataset, val_dataset, loader, model


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


def herding(args):
    """Perform herding selection (Speech data)
    """
    train_dataset, val_dataset, loader, model = load_pretrained_herding(args)
    f_idx = 3

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

    train_transform = None
    train_dataset = TensorDataset(data, target, train_transform)

    return train_dataset, val_dataset


def load_data_path(args):
    """Load synthetic data (Speech data)
    """
    train_dataset, valid_dataset, mean, std = load_logspec(args)

    if args.slct_type == 'idc':
        data, target = torch.load(os.path.join(f'{args.save_dir}', 'data.pt'))
        if args.factor > 1:
            data, target = decode(args, data, target)

        train_dataset = TensorDataset(data, target)
        print("Load condensed data ", args.save_dir, data.shape)

    else:
        indices = randomselect(train_dataset, args.ipc, nclass=args.nclass)
        train_dataset = Subset(train_dataset, indices)
        print(f"Random select {args.ipc} data (total {len(indices)})")

    print("Training data shape: ", train_dataset[0][0].shape)
    os.makedirs('./results', exist_ok=True)
    save_log_spec('./results/test.png', torch.stack([d[0] for d in train_dataset]), mean, std)
    print()

    return train_dataset, valid_dataset


def test_data(args,
              train_loader,
              val_loader,
              test_resnet=False,
              model_fn=None,
              repeat=1,
              logger=print,
              num_val=4):
    """Test synthetic data (Speech data)
    """
    args.epoch_print_freq = args.epochs // num_val

    if model_fn is None:
        model_fn_ls = [define_model]
        if test_resnet:
            model_fn_ls = [resnet10_bn]
    else:
        model_fn_ls = [model_fn]

    for model_fn in model_fn_ls:
        best_acc_l = []
        acc_l = []
        for i in range(repeat):
            model = model_fn(args, args.nclass, logger=logger)
            best_acc, acc = train(args, model, train_loader, val_loader, logger=print)
            best_acc_l.append(best_acc)
            acc_l.append(acc)
        logger(
            f'Repeat {repeat} => Best, last acc: {np.mean(best_acc_l):.1f} {np.mean(acc_l):.1f}\n')


if __name__ == '__main__':
    from argument import args
    import torch.backends.cudnn as cudnn
    import numpy as np

    cudnn.benchmark = True
    args.net_type = 'convnet'
    args.depth = 4
    args.size = 64

    base_path = "./results"
    if args.slct_type == 'idc':
        if args.name == '':
            if args.factor > 1:
                init = 'mix'
            else:
                init = 'random'
            name = f'speech/conv4in_grad_l1_nd500_niter1000_factor{args.factor}_none_lr0.003_{init}'
        else:
            name = args.name
        path_list = [f'{name}_ipc{ipc}' for ipc in [args.ipc]]
    else:
        path_list = ['']

    for p in path_list:
        args.save_dir = os.path.join(base_path, p)

        if args.slct_type == 'herding':
            train_dataset, val_dataset = herding(args)
        else:
            train_dataset, val_dataset = load_data_path(args)

        train_loader = MultiEpochsDataLoader(train_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=True,
                                             num_workers=args.workers if args.augment else 0,
                                             persistent_workers=args.augment > 0)
        val_loader = MultiEpochsDataLoader(val_dataset,
                                           batch_size=args.batch_size // 2,
                                           shuffle=False,
                                           persistent_workers=True,
                                           num_workers=4)

        test_data(args, train_loader, val_loader, repeat=args.repeat, test_resnet=False)
