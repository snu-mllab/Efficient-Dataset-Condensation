import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
from misc import utils
from data import TensorDataset
from data import ClassDataLoader, ClassMemDataLoader, MultiEpochsDataLoader
from train import define_model, train_epoch
from test import test_data
from misc.augment import DiffAug
from math import ceil
from transform_wav import LoadAudio, FixAudioLength
from transform_spec import ToSTFT
from speech import MiniSpeechCommandsDataset, resize_log_spectrograms, save_log_spec


class Synthesizer():
    def __init__(self, args, nclass, nchannel, hs, ws, device='cuda'):
        self.ipc = args.ipc
        self.nclass = nclass
        self.nchannel = nchannel
        self.size = (hs, ws)
        self.device = device

        self.data = torch.randn(size=(self.nclass * self.ipc, self.nchannel, hs, ws),
                                dtype=torch.float,
                                requires_grad=True,
                                device=self.device)
        self.targets = torch.tensor([np.ones(self.ipc) * i for i in range(nclass)],
                                    dtype=torch.long,
                                    requires_grad=False,
                                    device=self.device).view(-1)
        self.cls_idx = [[] for _ in range(self.nclass)]
        for i in range(self.data.shape[0]):
            self.cls_idx[self.targets[i]].append(i)

        print("\nDefine synthetic data: ", self.data.shape)

        self.factor = max(1, args.factor)
        self.decode_type = args.decode_type
        self.resize = nn.Upsample(size=self.size, mode='bilinear')
        print(f"Factor: {self.factor} ({self.decode_type})")

    def init(self, loader, init_type='noise'):
        if init_type == 'random':
            print("Random initialize synset")
            for c in range(self.nclass):
                img, _ = loader.class_sample(c, self.ipc)
                self.data.data[self.ipc * c:self.ipc * (c + 1)] = img.data.to(self.device)

        elif init_type == 'mix':
            print("Mixed initialize synset")
            for c in range(self.nclass):
                img, _ = loader.class_sample(c, self.ipc * self.factor)
                img = img.data.to(self.device)

                s = self.size[0] // self.factor
                remained = self.size[0] % self.factor

                n = self.ipc
                w_loc = 0
                h = self.size[0]
                k = 0
                for i in range(self.factor):
                    w_r = s + 1 if i < remained else s
                    img_part = F.interpolate(img[k * n:(k + 1) * n], size=(h, w_r))
                    self.data.data[n * c:n * (c + 1), :, :, w_loc:w_loc + w_r] = img_part
                    w_loc += w_r
                    k += 1

        elif init_type == 'noise':
            pass

    def parameters(self):
        parameter_list = [self.data]
        return parameter_list

    def subsample(self, data, target, max_size=-1):
        if (data.shape[0] > max_size) and (max_size > 0):
            indices = np.random.permutation(data.shape[0])
            data = data[indices[:max_size]]
            target = target[indices[:max_size]]

        return data, target

    def decode_zoom(self, img, target, factor):
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
        data_dec = self.resize(cropped)
        target_dec = torch.cat([target for _ in range(n_crop)])

        return data_dec, target_dec

    def decode(self, data, target, bound=128):
        if self.factor > 1:
            data, target = self.decode_zoom(data, target, self.factor)

        return data, target

    def sample(self, c, max_size=128):
        idx_from = self.ipc * c
        idx_to = self.ipc * (c + 1)
        data = self.data[idx_from:idx_to]
        target = self.targets[idx_from:idx_to]

        data, target = self.decode(data, target, bound=max_size)
        data, target = self.subsample(data, target, max_size=max_size)
        return data, target

    def loader(self, args, augment=True):
        train_transform = None

        data_dec = []
        target_dec = []
        for c in range(self.nclass):
            idx_from = self.ipc * c
            idx_to = self.ipc * (c + 1)
            data = self.data[idx_from:idx_to].detach()
            target = self.targets[idx_from:idx_to].detach()
            data, target = self.decode(data, target)

            data_dec.append(data)
            target_dec.append(target)

        data_dec = torch.cat(data_dec)
        target_dec = torch.cat(target_dec)

        train_dataset = TensorDataset(data_dec.cpu(), target_dec.cpu(), train_transform)

        print("Decode condensed data: ", data_dec.shape)
        nw = 0 if not augment else args.workers
        train_loader = MultiEpochsDataLoader(train_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=True,
                                             num_workers=nw,
                                             persistent_workers=nw > 0)

        return train_loader

    def test(self, args, val_loader, logger, bench=True):
        loader = self.loader(args, args.augment)
        # Test on current model
        test_data(args, loader, val_loader, logger=logger)


def load_resized_data(args):
    path = os.path.join(args.data_dir, 'mini_speech_commands')

    valid_transform = transforms.Compose(
        [LoadAudio(), FixAudioLength(),
         ToSTFT(absolute=True),
         transforms.ToTensor()])

    train_dataset = MiniSpeechCommandsDataset(path, valid_transform, train=True)
    valid_dataset = MiniSpeechCommandsDataset(path, valid_transform, train=False)

    processed_train = resize_log_spectrograms(train_dataset, args.size)
    processed_valid = resize_log_spectrograms(valid_dataset, args.size)

    mean = processed_train.mean()
    std = processed_train.std()
    processed_train -= mean
    processed_train /= std
    train_dataset = TensorDataset(processed_train, torch.tensor(train_dataset.targets))

    processed_valid -= mean
    processed_valid /= std
    valid_dataset = TensorDataset(processed_valid, torch.tensor(valid_dataset.targets))

    min_val = processed_train.min()
    train_dataset.nclass = 8
    valid_dataset.nclass = 8

    val_loader = MultiEpochsDataLoader(valid_dataset,
                                       batch_size=args.batch_size,
                                       shuffle=False,
                                       num_workers=4)

    return train_dataset, val_loader, mean, std, min_val


def remove_aug(augtype, remove_aug):
    aug_list = []
    for aug in augtype.split("_"):
        if aug not in remove_aug.split("_"):
            aug_list.append(aug)

    return "_".join(aug_list)


def diffaug(args):
    aug_type = args.aug_type
    print("Augmentataion Matching: ", aug_type)
    augment = DiffAug(strategy=aug_type, batch=True)
    aug_batch = augment

    if args.mixup_net == 'cut':
        aug_type = remove_aug(aug_type, 'cutout')
    print("Augmentataion Net update: ", aug_type)
    augment_rand = DiffAug(strategy=aug_type, batch=False)
    aug_rand = augment_rand

    return aug_batch, aug_rand


def dist(x, y, method='mse'):
    if method == 'mse':
        dist_ = (x - y).pow(2).sum()
    elif method == 'l1':
        dist_ = (x - y).abs().sum()
    elif method == 'l1_mean':
        n_b = x.shape[0]
        dist_ = (x - y).abs().reshape(n_b, -1).mean(-1).sum()
    elif method == 'cos':
        x = x.reshape(x.shape[0], -1)
        y = y.reshape(y.shape[0], -1)
        dist_ = torch.sum(1 - torch.sum(x * y, dim=-1) /
                          (torch.norm(x, dim=-1) * torch.norm(y, dim=-1) + 1e-6))
    return dist_


def add_loss(loss_sum, loss):
    if loss_sum == None:
        return loss
    else:
        return loss_sum + loss


def matchloss(args, img_real, img_syn, lab_real, lab_syn, model):
    loss = None

    if args.match == 'feat':
        with torch.no_grad():
            feat_tg = model.get_feature(img_real, args.idx_from, args.idx_to)
        feat = model.get_feature(img_syn, args.idx_from, args.idx_to)

        for i in range(len(feat)):
            loss = add_loss(loss, dist(feat_tg[i].mean(0), feat[i].mean(0), method=args.metric))

    elif args.match == 'grad':
        criterion = nn.CrossEntropyLoss()

        output_real = model(img_real)
        loss_real = criterion(output_real, lab_real)
        g_real = torch.autograd.grad(loss_real, model.parameters())
        g_real = list((g.detach() for g in g_real))

        output_syn = model(img_syn)
        loss_syn = criterion(output_syn, lab_syn)
        g_syn = torch.autograd.grad(loss_syn, model.parameters(), create_graph=True)

        for i in range(len(g_real)):
            if (len(g_real[i].shape) == 1) and not args.bias:  # bias, normliazation
                continue
            if (len(g_real[i].shape) == 2) and not args.fc:
                continue

            loss = add_loss(loss, dist(g_real[i], g_syn[i], method=args.metric))

    return loss


def condense(args, logger, device='cuda'):
    # Define real dataset and loader
    trainset, val_loader, mean, std, min_val = load_resized_data(args)
    if args.load_memory:
        loader_real = ClassMemDataLoader(trainset, batch_size=args.batch_real)
    else:
        loader_real = ClassDataLoader(trainset,
                                      batch_size=args.batch_real,
                                      num_workers=args.workers,
                                      shuffle=True,
                                      pin_memory=True,
                                      drop_last=True)
    nclass = trainset.nclass
    nch, hs, ws = trainset[0][0].shape

    # Define syn dataset
    synset = Synthesizer(args, nclass, nch, hs, ws)
    synset.init(loader_real, init_type=args.init)
    save_log_spec(os.path.join(args.save_dir, 'init.png'), synset.data, mean, std)

    # Define augmentation function
    aug, aug_rand = diffaug(args)
    save_log_spec(os.path.join(args.save_dir, f'aug.png'),
                  aug(synset.sample(0, max_size=args.batch_syn_max)[0]), mean, std)

    if not args.test:
        synset.test(args, val_loader, logger, bench=False)

    # Data distillation
    optim_img = torch.optim.SGD(synset.parameters(), lr=args.lr_img, momentum=args.mom_img)

    ts = utils.TimeStamp(args.time)
    n_iter = args.niter * 100 // args.inner_loop
    it_log = n_iter // 50
    it_test = [n_iter // 10, n_iter // 5, n_iter // 2, n_iter]

    logger(f"\nStart condensing with {args.match} matching for {n_iter} iteration")
    for it in range(n_iter):
        model = define_model(args, nclass).to(device)
        model.train()
        lr = 3e-4  # use small lr for update networks on full training set
        optim_net = optim.SGD(model.parameters(),
                                lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
        criterion = nn.CrossEntropyLoss()
        if args.early > 0:
            for _ in range(args.early):
                train_epoch(args,
                            loader_real,
                            model,
                            criterion,
                            optim_net,
                            aug=aug_rand,
                            mixup=args.mixup_net)

        loss_total = 0
        synset.data.data = torch.clamp(synset.data.data, min=min_val)
        for ot in range(args.inner_loop):
            ts.set()

            # Update synset
            for c in range(nclass):
                img, lab = loader_real.class_sample(c)
                img_syn, lab_syn = synset.sample(c, max_size=args.batch_syn_max)
                ts.stamp("data")

                n = img.shape[0]
                img_aug = aug(torch.cat([img, img_syn]))
                ts.stamp("aug")

                loss = matchloss(args, img_aug[:n], img_aug[n:], lab, lab_syn, model)
                loss_total += loss.item()
                ts.stamp("loss")

                optim_img.zero_grad()
                loss.backward()
                optim_img.step()
                ts.stamp("backward")

            # Net update
            if args.n_data > 0:
                for _ in range(args.net_epoch):
                    train_epoch(args,
                                loader_real,
                                model,
                                criterion,
                                optim_net,
                                n_data=args.n_data,
                                aug=aug_rand,
                                mixup=args.mixup_net)
            ts.stamp("net update")

            if (ot + 1) % 10 == 0:
                ts.flush()

        # Logging
        if it % it_log == 0:
            logger(
                f"{utils.get_time()} (Iter {it:3d}) loss: {loss_total/nclass/args.inner_loop:.1f}")
        if (it + 1) in it_test:
            save_log_spec(os.path.join(args.save_dir, f'img{it+1}.png'), synset.data, mean, std)
            torch.save(
                [synset.data.detach().cpu(), synset.targets.cpu()],
                os.path.join(args.save_dir, f'data.pt'))
            print("img and data saved!")

            if not args.test:
                synset.test(args, val_loader, logger)


if __name__ == '__main__':
    import shutil
    from misc.utils import Logger
    from argument import args
    import torch.backends.cudnn as cudnn
    import json

    cudnn.benchmark = True
    if args.seed > 0:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    os.makedirs(args.save_dir, exist_ok=True)

    cur_file = os.path.join(os.getcwd(), __file__)
    shutil.copy(cur_file, args.save_dir)

    logger = Logger(args.save_dir)
    logger(f"Save dir: {args.save_dir}")
    with open(os.path.join(args.save_dir, 'args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    condense(args, logger)