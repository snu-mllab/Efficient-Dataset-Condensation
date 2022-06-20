"""Transforms on the short time fourier transforms of wav samples."""
__author__ = 'Erdene-Ochir Tuguldur'

import random

import numpy as np
import librosa
from torch.utils.data import Dataset

SAMPLE_RATE = 16000
N_FFT = 255
HOP_LENGTH = 128


def should_apply_transform(prob=0.5):
    """Transforms are only randomly applied with the given probability."""
    return random.random() < prob


class ToSTFT(object):
    """Applies on an audio the short time fourier transform."""
    def __init__(self, n_fft=N_FFT, hop_length=HOP_LENGTH, absolute=False):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.abs = absolute

    def __call__(self, samples):
        stft = librosa.stft(samples, n_fft=self.n_fft, hop_length=self.hop_length)
        if self.abs:
            stft = np.abs(stft)
        return stft


class StretchAudioOnSTFT(object):
    """Stretches an audio on the frequency domain."""
    def __init__(self, max_scale=0.2):
        self.max_scale = max_scale

    def __call__(self, stft):
        if not should_apply_transform():
            return stft

        scale = random.uniform(-self.max_scale, self.max_scale)
        stft = librosa.core.phase_vocoder(stft, 1 + scale, hop_length=HOP_LENGTH)
        return stft


class TimeshiftAudioOnSTFT(object):
    """A simple timeshift on the frequency domain without multiplying with exp."""
    def __init__(self, max_shift=8):
        self.max_shift = max_shift

    def __call__(self, stft):
        if not should_apply_transform():
            return stft

        shift = random.randint(-self.max_shift, self.max_shift)
        a = -min(0, shift)
        b = max(0, shift)
        stft = np.pad(stft, ((0, 0), (a, b)), "constant")
        if a == 0:
            stft = stft[:, b:]
        else:
            stft = stft[:, 0:-a]
        return stft


class FixSTFTDimension(object):
    """Either pads or truncates in the time axis on the frequency domain, applied after stretching, time shifting etc."""
    def __call__(self, stft):
        t_len = stft.shape[1]
        orig_t_len = stft.shape[1]
        if t_len > orig_t_len:
            stft = stft[:, 0:orig_t_len]
        elif t_len < orig_t_len:
            stft = np.pad(stft, ((0, 0), (0, orig_t_len - t_len)), "constant")

        return stft


class ToMelSpectrogramFromSTFT(object):
    """Creates the mel spectrogram from the short time fourier transform of a file. The result is a 32x32 matrix."""
    def __init__(self, n_mels=32):
        self.n_mels = n_mels

    def __call__(self, stft):
        n_fft = N_FFT
        mel_basis = librosa.filters.mel(SAMPLE_RATE, n_fft, self.n_mels)
        s = np.dot(mel_basis, np.abs(stft)**2.0)
        mel = librosa.power_to_db(s, ref=np.max)
        return mel


class AudioFromSTFT(object):
    """Inverse short time fourier transform."""
    def __call__(self, stft):
        samples = librosa.core.istft(stft)
        return samples


class AudioFromMSTFT(object):
    """Inverse short time fourier transform."""
    def __call__(self, stft):
        samples = librosa.griffinlim(stft, win_length=N_FFT, hop_length=HOP_LENGTH)
        return samples