"""Load Google Speech Command dataset."""

from typing import Dict, List, Tuple
import hashlib
import logging
import os
import re

import torch

Index = Dict[str, List[str]]

LOGGER = logging.getLogger('spotter_train')
MAX_NUM_WAVS_PER_CLASS = 2**27 - 1  # ~134M

SILENCE = '_silence_'
BACKGROUND_FOLDER = '_background_noise_'


class GoogleSpeechCommandsDataset(torch.utils.data.IterableDataset):
    pass


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


def which_set(fname: str, dev_percentage: float, test_percentage: float) -> int:
    """Return 0 for train, 1 for dev, 2 for test.
    Reason: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/speech_commands/input_data.py#L70
    """
    base_name = os.path.basename(fname)
    hash_name = re.sub(r'_nohash_.*$', '', base_name)
    hash_name_hashed = hashlib.sha1(hash_name.encode('UTF-8')).hexdigest()
    percentage_hash = ((int(hash_name_hashed, 16) % (MAX_NUM_WAVS_PER_CLASS + 1)) * (100.0 / MAX_NUM_WAVS_PER_CLASS))
    if percentage_hash < dev_percentage:
        return 1
    if percentage_hash < dev_percentage + test_percentage:
        return 2
    return 0


def split_index(index: Index, dev_percentage: float, test_percentage: float) -> Tuple[Index, Index, Index]:
    """Split whole index into train/dev/test."""
    result = {}, {}, {}
    for label, fnames in index.items():
        for cresult in result:
            cresult[label] = []
        for fname in fnames:
            result[which_set(fname, dev_percentage, test_percentage)][label].append(fname)
    return result
