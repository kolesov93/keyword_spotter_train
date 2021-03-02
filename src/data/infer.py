import torch
import torchaudio

from .common import SAMPLE_RATE

from typing import Dict

class InferenceDataset(torch.utils.data.IterableDataset):

    def __init__(self, u2path: Dict[str, str], samples_shift: int = 8000):
        self._u2path = u2path
        self._samples_shift = samples_shift

    def __iter__(self):
        for uttid, path in self._u2path.items():
            waveform, sample_rate = torchaudio.load(path)
            if sample_rate != SAMPLE_RATE:
                raise ValueError(f'{path} has sample rate {sample_rate}, not {SAMPLE_RATE}')
            data = waveform[0].numpy()

            for shift in range(0, data.shape[0], self._samples_shift):
                if shift + SAMPLE_RATE > data.shape[0]:
                    break
                yield {
                    'uttid': uttid,
                    'shift': shift,
                    'data': data[shift: shift + SAMPLE_RATE]
                }

