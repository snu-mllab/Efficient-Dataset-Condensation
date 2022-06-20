"""Transforms on raw wav samples."""
__author__ = 'Yuan Xu'

import random
import numpy as np
import librosa
import torch

SAMPLE_RATE = 16000


def should_apply_transform(prob=0.5):
    """Transforms are only randomly applied with the given probability."""
    return random.random() < prob


class LoadAudio(object):
    """Loads an audio into a numpy array."""
    def __init__(self, sample_rate=SAMPLE_RATE):
        self.sample_rate = sample_rate

    def __call__(self, path):
        if path:
            samples, sample_rate = librosa.load(path, self.sample_rate)
        else:
            # silence
            sample_rate = self.sample_rate
            samples = np.zeros(sample_rate, dtype=np.float32)

        return samples


class FixAudioLength(object):
    """Either pads or truncates an audio into a fixed length."""
    def __init__(self, time=1):
        self.time = time

    def __call__(self, data):
        length = int(self.time * SAMPLE_RATE)
        if length < len(data):
            data = data[:length]
        elif length > len(data):
            data = np.pad(data, (0, length - len(data)), "constant")
        return data


class ChangeAmplitude(object):
    """Changes amplitude of an audio randomly."""
    def __init__(self, amplitude_range=(0.7, 1.1)):
        self.amplitude_range = amplitude_range

    def __call__(self, data):
        if not should_apply_transform():
            return data

        data = data * random.uniform(*self.amplitude_range)
        return data


class ChangeSpeedAndPitchAudio(object):
    """Change the speed of an audio. This transform also changes the pitch of the audio."""
    def __init__(self, max_scale=0.2):
        self.max_scale = max_scale

    def __call__(self, samples):
        if not should_apply_transform():
            return samples

        scale = random.uniform(-self.max_scale, self.max_scale)
        speed_fac = 1.0 / (1 + scale)
        data = np.interp(np.arange(0, len(samples), speed_fac), np.arange(0, len(samples)),
                         samples).astype(np.float32)
        return data


class StretchAudio(object):
    """Stretches an audio randomly."""
    def __init__(self, max_scale=0.2):
        self.max_scale = max_scale

    def __call__(self, data):
        if not should_apply_transform():
            return data

        scale = random.uniform(-self.max_scale, self.max_scale)
        data = librosa.effects.time_stretch(data, 1 + scale)
        return data


class TimeshiftAudio(object):
    """Shifts an audio randomly."""
    def __init__(self, max_shift_seconds=0.2):
        self.max_shift_seconds = max_shift_seconds

    def __call__(self, data):
        if not should_apply_transform():
            return data

        samples = data
        max_shift = (SAMPLE_RATE * self.max_shift_seconds)
        shift = random.randint(-max_shift, max_shift)
        a = -min(0, shift)
        b = max(0, shift)
        samples = np.pad(samples, (a, b), "constant")
        data = samples[:len(samples) - a] if a else samples[b:]
        return data


class ToMelSpectrogram(object):
    """Creates the mel spectrogram from an audio. The result is a 32x32 matrix."""
    def __init__(self, n_mels=32):
        self.n_mels = n_mels

    def __call__(self, samples):
        samples = samples
        s = librosa.feature.melspectrogram(samples, sr=SAMPLE_RATE, n_mels=self.n_mels)
        data = librosa.power_to_db(s, ref=np.max)
        return data


class ToTensor(object):
    """Converts into a tensor."""
    def __init__(self, np_name, tensor_name, normalize=None):
        self.np_name = np_name
        self.tensor_name = tensor_name
        self.normalize = normalize

    def __call__(self, data):
        tensor = torch.FloatTensor(data)
        if self.normalize is not None:
            mean, std = self.normalize
            tensor -= mean
            tensor /= std
        return tensor