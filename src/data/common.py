import enum
import numpy as np

from typing import Optional

class DatasetTag(enum.IntEnum):
    TRAIN = 0
    DEV = 1
    TEST = 2

SAMPLE_RATE = 16000
NOISE_COEF = 0.1


def ensure_duration(audio: np.array) -> np.array:
    samples = SAMPLE_RATE  # exactly one second

    if len(audio) == samples:
        return audio

    if len(audio) > samples:
        to_remove = len(audio) - samples
        left = to_remove // 2
        return audio[left: left + samples]

    to_pad = samples - len(audio)
    left = to_pad // 2
    right = to_pad - left

    return np.pad(audio, (left, right), 'constant')


def get_random_snippet(audio: np.array, samples: Optional[int], rnd: Optional[np.random.RandomState]) -> np.array:
    """Get a random subsample of audio."""
    if samples is None:
        samples = SAMPLE_RATE # exactly one second
    if rnd is None:
        rnd = np.random.RandomState()

    if len(audio) < samples:
        raise ValueError(f'Number of samples is to small for a sample')

    start = rnd.choice(len(audio) - samples + 1)
    return audio[start: start + samples]


def add_noise(audio: np.array, noise: np.array, rnd: Optional[np.random.RandomState]) -> np.array:
    noise = rnd.random() * NOISE_COEF * get_random_snippet(noise, len(audio), rnd)
    return np.clip(audio + noise, -1., 1.)
