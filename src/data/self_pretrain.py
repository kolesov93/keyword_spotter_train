import torch
import torchaudio
import numpy as np

from .common import SAMPLE_RATE, ensure_duration

from typing import Dict, Optional, List


class SelfPretrainDataset(torch.utils.data.IterableDataset):

    def __init__(self, paths: List[str], seed: int, xerox: Optional[int] = None, augument: bool = True, level: int = 1):
        """
        - paths - paths to .wav files
        - seed - random seed
        - xerox - how many times to duplicates each sample. If None, generate infinite set.
        - augument - perform augugmentation (False is useful for test)
        if not None, generate fixed set (for validation and test)
        """
        self._paths = paths
        self._rnd = np.random.RandomState(seed=seed)
        self._cache = {}
        self._fixed_set = None
        self._level = level

        if xerox is not None:
            self._prepare_fixed_set(xerox)

    def _read_audio(self, fname):
        if fname in self._cache:
            return self._cache[fname]
        waveform, sample_rate = torchaudio.load(fname)
        if sample_rate != SAMPLE_RATE:
            raise ValueError(f'{fname} has sample rate {sample_rate}, not {SAMPLE_RATE}')
        waveform = np.array([ensure_duration(waveform[0].cpu().detach().numpy())])
        waveform = torch.from_numpy(waveform)
        data = torchaudio.compliance.kaldi.fbank(
            waveform,
            num_mel_bins=80,
            frame_shift=10,
            frame_length=25
        )
        self._cache[fname] = data
        return data

    def _augument(self, spectrogram):
        spectrogram = spectrogram.detach().clone()
        if self._level == 1:
            freq_times = 1
            time_times = 1
            max_freqs = 25
            max_times = 25
        elif self._level == 2:
            freq_times = 2
            time_times = 2
            max_freqs = 15
            max_times = 15
        elif self._level == 3:
            freq_times = 1
            time_times = 1
            max_freqs = 40
            max_times = 40

        for _ in range(freq_times):
            f = self._rnd.randint(low=1, high=max_freqs)
            f0 = self._rnd.randint(low=0, high=spectrogram.shape[1] - f)
            spectrogram[:, f0:f0+f] = 0.

        for _ in range(time_times):
            t = self._rnd.randint(low=1, high=max_times)
            t0 = self._rnd.randint(low=0, high=spectrogram.shape[0] - t)
            spectrogram[t0:t0+t, :] = 0.

        return spectrogram

    def _prepare_fixed_set(self, xerox):
        self._fixed_set = []
        for i, path in enumerate(self._paths):
            original_spectrogram = self._read_audio(path)
            for _ in range(xerox):
                if self._augument:
                    spectrogram = self._augument(original_spectrogram)
                else:
                    spectrogram = original_spectrogram
                self._fixed_set.append({
                    'data': spectrogram,
                    'label': i
                })

    def __iter__(self):
        if self._fixed_set:
            yield from self._fixed_set
            return

        while True:
            label = self._rnd.randint(0, len(self._paths))
            spectrogram = self._read_audio(self._paths[label])
            if self._augument:
                spectrogram = self._augument(spectrogram)
            yield {
                'data': spectrogram,
                'label': label
            }


def collate_fn(samples):
    x, y = [], []
    for sample in samples:
        x.append(sample['data'].reshape(1, -1, 80))
        y.append(sample['label'])
    x = torch.cat(x, 0)
    y = np.array(y, dtype=np.int64)
    return x, torch.from_numpy(y)
