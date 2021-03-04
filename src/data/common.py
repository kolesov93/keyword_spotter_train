import enum
import numpy as np

class DatasetTag(enum.IntEnum):
    TRAIN = 0
    DEV = 1
    TEST = 2

SAMPLE_RATE = 16000


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

