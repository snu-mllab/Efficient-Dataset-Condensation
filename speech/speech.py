import os
from torch.utils.data import Dataset
import torch
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from transform_wav import LoadAudio, FixAudioLength
from transform_spec import ToSTFT
from data import TensorDataset

CLASSES = ['yes', 'no', 'go', 'stop', 'left', 'right', 'up', 'down']


class MiniSpeechCommandsDataset(Dataset):
    """Google mini speech commands dataset.
    See for more information: https://www.tensorflow.org/tutorials/audio/simple_audio
    """
    def __init__(self, folder, transform=None, train=True, classes=CLASSES):
        all_classes = [
            d for d in os.listdir(folder)
            if os.path.isdir(os.path.join(folder, d)) and not d.startswith('_')
        ]

        class_to_idx = {classes[i]: i for i in range(len(classes))}

        self.data = []
        self.targets = []
        for c in CLASSES:
            d = os.path.join(folder, c)
            target = class_to_idx[c]
            file_names = sorted(os.listdir(d))

            for i, f in enumerate(file_names):
                if train:
                    if i % 8 != 0:
                        # if i < 875:
                        path = os.path.join(d, f)
                        self.data.append(path)
                        self.targets.append(target)
                else:
                    if i % 8 == 0:
                        path = os.path.join(d, f)
                        self.data.append(path)
                        self.targets.append(target)

        self.classes = classes
        self.transform = transform
        self.nclass = len(all_classes)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]
        target = self.targets[index]

        if self.transform is not None:
            data = self.transform(data)

        return data, target


class SpeechCommandsDataset(Dataset):
    """Google speech commands dataset.
    """
    def __init__(self, folder, transform=None, train=True, classes=CLASSES):
        all_classes = [
            d for d in os.listdir(folder)
            if os.path.isdir(os.path.join(folder, d)) and not d.startswith('_')
        ]

        class_to_idx = {classes[i]: i for i in range(len(classes))}

        self.data = []
        self.targets = []
        for c in CLASSES:
            d = os.path.join(folder, c)
            target = class_to_idx[c]
            file_names = sorted(os.listdir(d))

            for i, f in enumerate(file_names):
                path = os.path.join(d, f)
                self.data.append(path)
                self.targets.append(target)

        self.classes = classes
        self.transform = transform
        self.nclass = len(all_classes)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]
        target = self.targets[index]

        if self.transform is not None:
            data = self.transform(data)

        return data, target


def load_logspec(args):
    """Load normalized log-spectrogram
    """
    path = os.path.join(args.data_dir, 'mini_speech_commands')

    valid_transform = transforms.Compose([
        LoadAudio(),
        FixAudioLength(),
        ToSTFT(absolute=True),
        transforms.ToTensor(),
    ])

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

    return train_dataset, valid_dataset, mean, std


def resize_log_spectrograms(dataset, size=64, eps=1e-3):
    processed = []
    for i, (spec, _) in enumerate(dataset):
        processed.append(spec)
    processed = torch.stack(processed)

    processed = torch.nn.functional.pad(processed, (1, 2))
    if size < 128:
        processed = torch.nn.functional.interpolate(processed, size=(size, size), mode='bilinear')
    print("Data shape:", processed.shape)

    visualize_spec(processed[0])
    processed = torch.log(processed + eps)

    return processed


def visualize_spec(spec, name='spec'):
    spec = spec.permute(1, 2, 0).flip(0)
    plt.imshow(spec.numpy(), cmap='jet', vmin=0)
    plt.savefig(f'./{name}.png')


def save_log_spec(name, spec, mean, std, ncol=10):
    spec = spec.detach().cpu() * std + mean
    spec = torch.exp(spec)
    spec = torch.nn.functional.pad(spec, (1, 1, 1, 1))
    n = len(spec)
    spec = spec / spec.reshape(n, -1).max(-1)[0].reshape(n, 1, 1, 1)
    spec = spec.permute(0, 2, 3, 1).flip(1)

    remain = len(spec) % ncol
    if remain > 0:
        blank = [torch.zeros_like(spec[0]) for _ in range(remain)]
        blank = torch.stack(blank)
        spec = torch.cat([spec, blank])

    rows = []
    for k in range(len(spec) // ncol):
        row = [spec[j + k * ncol] for j in range(ncol)]
        row = torch.cat(row, dim=1)
        rows.append(row)
    img_full = torch.cat(rows, dim=0)

    plt.imshow(img_full.numpy(), cmap='jet', vmin=0)
    plt.savefig(f'{name}')


if __name__ == '__main__':
    from argument import args
    from data import MultiEpochsDataLoader
    from test import test_data

    use_gpu = torch.cuda.is_available()
    if use_gpu:
        torch.backends.cudnn.benchmark = True

    train_dataset, valid_dataset, mean, std = load_logspec(args)
    train_loader = MultiEpochsDataLoader(train_dataset, batch_size=args.batch_size, num_workers=4)
    val_loader = MultiEpochsDataLoader(valid_dataset,
                                       batch_size=args.batch_size,
                                       shuffle=False,
                                       num_workers=4)

    args.epochs = 180
    num_val = args.epochs
    test_data(args, train_loader, val_loader, num_val=num_val)
