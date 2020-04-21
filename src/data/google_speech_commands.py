"""Load Google Speech Command dataset."""

from typing import Dict, List, Tuple
import hashlib
import logging
import math
import os
import re

import torch
import librosa
import numpy as np

from .common import DatasetTag

Index = Dict[str, List[str]]

LOGGER = logging.getLogger('spotter_train')
MAX_NUM_WAVS_PER_CLASS = 2**27 - 1  # ~134M

SILENCE = '_silence_'
UNKNOWN = '_unknown_'

BACKGROUND_FOLDER = '_background_noise_'

SILENCE_PROB = 0.1
UNKNOWN_PROB = 0.1

SILENCE_LABEL = 0
UNKNOWN_LABEL = 1

SAMPLE_RATE = 16000

NOISE_PROB = 0.7
NOISE_COEF = 0.1


def _get_samples(index: Index, wanted_words: List[str]) -> List[Tuple[str, int]]:
    """Get only wanted words: (fname, label).
    Note: first two labels are reserved for silence and unknown"""
    result = []
    for label, word in enumerate(wanted_words):
        if word not in index:
            raise ValueError(f'No samples for "{word}" in index')
        for fname in index[word]:
            result.append((fname, label + 2))
    np.random.RandomState(seed=1993).shuffle(result)
    return result


def _get_bg_samples(index: Index) -> List[str]:
    if (BACKGROUND_FOLDER not in index) or (not index[BACKGROUND_FOLDER]):
        raise ValueError(f'No background samples')
    return index[BACKGROUND_FOLDER]


def _get_unknown_samples(index: Index, wanted_words: List[str]) -> List[Tuple[str, int]]:
    """Get not wanted words and not BACKGROUND_FOLDER."""
    wanted_words = set(wanted_words)
    result = []
    for label, fnames in index.items():
        if label in wanted_words or label == BACKGROUND_FOLDER:
            continue
        result.extend(fnames)
    np.random.RandomState(seed=1993).shuffle(result)
    return result


def _get_worker_slice(a):
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is None:
        return a
    per_worker = int(math.ceil(len(a) / worker_info.num_workers))
    worker_id = worker_info.id
    start = worker_id * per_worker
    end = min(start + per_worker, len(a))
    return a[start: end]


def _get_seed_for_worker() -> int:
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is None:
        return 1992
    return 1993 + worker_info.id


def _get_label(label: int, wanted_words: List[str]) -> str:
    if label == SILENCE_LABEL:
        return SILENCE
    if label == UNKNOWN_LABEL:
        return UNKNOWN
    if label - 2 < 0 or label - 2 >= len(wanted_words):
        raise ValueError(f'Unknown label {label}')
    return wanted_words[label - 2]


def _ensure_duration(audio: np.array) -> np.array:
    samples = SAMPLE_RATE # exactly one second

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


def _get_snippet(audio: np.array, rnd: np.random.RandomState) -> np.array:
    """Get a random subsample of audio."""
    samples = SAMPLE_RATE # exactly one second

    if len(audio) < samples:
        raise ValueError(f'Number of samples is to small for a sample')

    start = rnd.choice(len(audio) - samples + 1)
    return audio[start: start + samples]


class GoogleSpeechCommandsDataset(torch.utils.data.IterableDataset):

    def __init__(self, index: Index, wanted_words: List[str], tag: DatasetTag):
        self._samples = _get_samples(index, wanted_words)
        self._bg_samples = _get_bg_samples(index)
        self._unknown_samples = _get_unknown_samples(index, wanted_words)

        self._tag = tag

        self._cache = {}

    def _read_audio(self, fname):
        if fname in self._cache:
            return self._cache[fname]
        data = librosa.core.load(fname, sr=SAMPLE_RATE)[0]
        self._cache[fname] = data
        return data

    def _get_bg_snippet(self, rnd: np.random.RandomState) -> np.array:
        fname = rnd.choice(self._bg_samples)
        return _get_snippet(self._read_audio(fname), rnd)

    def _add_noise(self, audio: np.array, rnd: np.random.RandomState) -> np.array:
        noise = rnd.random() * NOISE_COEF * self._get_bg_snippet(rnd)
        return np.clip(audio + noise, -1., 1.)

    def _get_unknown_sample(self, unknown_samples: List[str], rnd: np.random.RandomState, add_noise: bool):
        fname = rnd.choice(unknown_samples)
        data = _ensure_duration(self._read_audio(fname))
        if add_noise:
            data = self._add_noise(data, rnd)
        return {'data': data, 'label': UNKNOWN_LABEL}

    def _get_silence_sample(self, rnd: np.random.RandomState):
        data = rnd.random() * NOISE_COEF * self._get_bg_snippet(rnd)
        return {'data': data, 'label': SILENCE_LABEL}

    def _get_command_sample(self, samples: List[Tuple[str, int]], rnd: np.random.RandomState, add_noise: bool):
        fname, label = samples[rnd.choice(len(samples))]
        data = _ensure_duration(self._read_audio(fname))
        if add_noise:
            data = self._add_noise(data, rnd)
        return {'data': data, 'label': label}

    def __iter__(self):
        samples = _get_worker_slice(self._samples)
        unknown_samples = _get_worker_slice(self._unknown_samples)

        rnd = np.random.RandomState(seed=_get_seed_for_worker())
        for _ in range(20):
            coin = rnd.random()

            add_noise = rnd.random() < NOISE_PROB
            if coin < UNKNOWN_PROB:
                yield self._get_unknown_sample(unknown_samples, rnd, add_noise)
            elif coin < UNKNOWN_PROB + SILENCE_PROB:
                yield self._get_silence_sample(rnd)
            else:
                yield self._get_command_sample(samples, rnd, add_noise)


def get_index(folder: str) -> Index:
    """For each label returns list of file paths."""
    result = {}

    folder = os.path.abspath(folder)
    for subfolder_name in os.listdir(folder):
        subfolder = os.path.join(folder, subfolder_name)
        if not os.path.isdir(subfolder):
            continue

        for fname in os.listdir(subfolder):
            if not fname.endswith('.wav'):
                LOGGER.warning('Skipping file %s', fname)
                continue

            fname = os.path.join(subfolder, fname)
            if subfolder_name in result:
                result[subfolder_name].append(fname)
            else:
                result[subfolder_name] = [fname]

    return result


def which_set(fname: str, dev_percentage: float, test_percentage: float) -> DatasetTag:
    """
    See https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/speech_commands/input_data.py#L70
    """
    base_name = os.path.basename(fname)
    hash_name = re.sub(r'_nohash_.*$', '', base_name)
    hash_name_hashed = hashlib.sha1(hash_name.encode('UTF-8')).hexdigest()
    percentage_hash = ((int(hash_name_hashed, 16) % (MAX_NUM_WAVS_PER_CLASS + 1)) * (100.0 / MAX_NUM_WAVS_PER_CLASS))
    if percentage_hash < dev_percentage:
        return DatasetTag.DEV
    if percentage_hash < dev_percentage + test_percentage:
        return DatasetTag.TEST
    return DatasetTag.TRAIN


def split_index(index: Index, dev_percentage: float, test_percentage: float) -> Tuple[Index, Index, Index]:
    """Split whole index into train/dev/test."""
    result = {}, {}, {}
    for label, fnames in index.items():
        for cresult in result:
            cresult[label] = []
        for fname in fnames:
            if label == BACKGROUND_FOLDER:
                # this is a potential leak, but I'm repoducing https://github.com/castorini/honk/blob/master/utils/model.py#L339
                for cresult in result:
                    cresult[label].append(fname)
            else:
                result[which_set(fname, dev_percentage, test_percentage)][label].append(fname)
    return result
