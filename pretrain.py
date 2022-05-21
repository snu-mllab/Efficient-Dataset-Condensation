import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from data import ClassDataLoader, ClassMemDataLoader
from train import define_model, train_epoch, validate
from condense import load_resized_data, diffaug


def pretrain(args, logger, device='cuda'):
    trainset, val_loader = load_resized_data(args)
    if args.load_memory:
        loader_real = ClassMemDataLoader(trainset, batch_size=args.batch_real)
    else:
        loader_real = ClassDataLoader(trainset,
                                      batch_size=args.batch_real,
                                      num_workers=args.workers,
                                      shuffle=True,
                                      pin_memory=True,
                                      drop_last=False)
    nclass = trainset.nclass
    _, aug_rand = diffaug(args)

    model = define_model(args, nclass).to(device)
    model.train()
    optim_net = optim.SGD(model.parameters(),
                          args.lr,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    epoch_max = args.pt_from + args.pt_num
    print(f"Start training for {epoch_max} epochs")
    for epoch in range(1, epoch_max):
        top1, _, loss = train_epoch(args,
                                    loader_real,
                                    model,
                                    criterion,
                                    optim_net,
                                    epoch=epoch,
                                    aug=aug_rand,
                                    mixup=args.mixup_net,
                                    logger=logger)
        top1_val, _, _ = validate(args, val_loader, model, criterion, epoch, logger=logger)
        print(f"[Epoch {epoch}] Train acc: {top1:.1f} (loss: {loss:.3f}), Val acc: {top1_val:.1f}")

        if epoch >= args.pt_from:
            ckpt_path = os.path.join(args.save_dir, f'checkpoint{epoch}.pth.tar')
            torch.save(model.state_dict(), ckpt_path)


if __name__ == '__main__':
    import shutil
    from misc.utils import Logger
    from argument import args
    import torch.backends.cudnn as cudnn

    assert args.pt_from > 0, "set args.pt_from positive! (epochs for pretraining)"

    cudnn.benchmark = True
    if args.seed > 0:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    args.save_dir = f"./pretrained/{args.datatag}/{args.modeltag}{args.tag}_{args.seed}"
    os.makedirs(args.save_dir, exist_ok=True)

    cur_file = os.path.join(os.getcwd(), __file__)
    shutil.copy(cur_file, args.save_dir)

    logger = Logger(args.save_dir)
    logger(f"Save dir: {args.save_dir}")
    pretrain(args, logger)