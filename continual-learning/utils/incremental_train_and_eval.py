import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.augment import DiffAug, random_indices, rand_bbox
import time


def incremental_train_and_eval(
    epochs,
    tg_model,
    ref_model,
    tg_optimizer,
    tg_lr_scheduler,
    trainloader,
    testloader,
    iteration,
    start_iteration,
    T,
    beta,
    strong_aug,
    mix_p,
    is_dsa,
):
    print(f"mix_p : {mix_p}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tg_model = tg_model.to(device)

    aug_fn = DiffAug(
        strategy="color_crop_cutout_flip_scale_rotate"
        if is_dsa
        else "color_crop_flip_scale_rotate",
        single=is_dsa,
    )
    criterion = nn.CrossEntropyLoss().cuda()

    quad = int(epochs / 4)

    print(trainloader.dataset)
    t1 = time.time()

    if iteration > start_iteration:
        ref_model.eval()
        num_old_classes = ref_model.fc.out_features
    for epoch in range(1, epochs + 1):
        # train
        tg_model.train()
        train_loss = 0
        train_loss1 = 0
        train_loss2 = 0
        correct = 0
        total = 0
        if epoch % quad == 0:
            print("\nEpoch: %d, LR: " % epoch, end="")
            print(tg_lr_scheduler.get_last_lr())

        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)

            tg_optimizer.zero_grad()
            if strong_aug:
                with torch.no_grad():
                    inputs = aug_fn(inputs)

                beta_param = 1.0
                r = np.random.rand(1)
                if r < mix_p:
                    # generate mixed sample
                    lam = np.random.beta(beta_param, beta_param)
                    rand_index = random_indices(targets)

                    target_b = targets[rand_index]
                    bbx1, bby1, bbx2, bby2 = rand_bbox(inputs.size(), lam)
                    inputs[:, :, bbx1:bbx2, bby1:bby2] = inputs[
                        rand_index, :, bbx1:bbx2, bby1:bby2
                    ]
                    ratio = 1 - (
                        (bbx2 - bbx1)
                        * (bby2 - bby1)
                        / (inputs.size()[-1] * inputs.size()[-2])
                    )

                    outputs = tg_model(inputs)
                    loss2 = criterion(outputs, targets) * ratio + criterion(
                        outputs, target_b
                    ) * (1.0 - ratio)
                else:
                    # compute output
                    outputs = tg_model(inputs)
                    loss2 = criterion(outputs, targets)
            else:
                outputs = tg_model(inputs)
                loss2 = criterion(outputs, targets)

            # print(f'Time until 1 : {time() - t1:2f} seconds')
            if iteration == start_iteration:
                loss1 = 0
            else:
                ref_outputs = ref_model(inputs)
                loss1 = (
                    nn.KLDivLoss()(
                        F.log_softmax(outputs[:, :num_old_classes] / T, dim=1),
                        F.softmax(ref_outputs.detach() / T, dim=1),
                    )
                    * T
                    * T
                    * beta
                    * num_old_classes
                )

            loss = loss1 + loss2
            loss.backward()
            tg_optimizer.step()

            train_loss += loss.item()
            if iteration > start_iteration:
                train_loss1 += loss1.item()
                train_loss2 += loss2.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        tg_lr_scheduler.step()

        if epoch % quad == 0:
            print(f"Train time : {time.time() - t1:.2f} seconds")
            if iteration == start_iteration:
                print(
                    "Train set: {}, Train Loss: {:.4f} Acc: {:.4f}".format(
                        len(trainloader),
                        train_loss / (batch_idx + 1),
                        100.0 * correct / total,
                    )
                )
            else:
                print(
                    "Train set: {}, Train Loss1: {:.4f}, Train Loss2: {:.4f},\
                    Train Loss: {:.4f} Acc: {:.4f}".format(
                        len(trainloader),
                        train_loss1 / (batch_idx + 1),
                        train_loss2 / (batch_idx + 1),
                        train_loss / (batch_idx + 1),
                        100.0 * correct / total,
                    )
                )

            # eval
            tg_model.eval()
            test_loss = 0
            correct = 0
            total = 0
            with torch.no_grad():
                for batch_idx, (inputs, targets) in enumerate(testloader):
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = tg_model(inputs)
                    loss = nn.CrossEntropyLoss()(outputs, targets)

                    test_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()

                    # progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    #    % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
            print(
                "Test set: {} Test Loss: {:.4f} Acc: {:.4f}".format(
                    len(testloader),
                    test_loss / (batch_idx + 1),
                    100.0 * correct / total,
                )
            )

    t2 = time.time()
    print()
    print("#################################################")
    print(f"Elapsed time for training is {t2 - t1} seconds")
    print("#################################################")
    print()
    return tg_model
