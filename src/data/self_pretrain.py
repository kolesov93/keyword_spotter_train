import torch
import torchaudio
import numpy as np
import logging
from dataclasses import dataclass

from .common import SAMPLE_RATE, ensure_duration

from typing import Dict, Optional, List

LOGGER = logging.getLogger('spotter_train')

@dataclass
class SpecAugumentOptions:
    max_times: int
    max_freqs: int
    freq_times: int
    time_times: int

SPECAUG_LEVELS = [
    SpecAugumentOptions(max_times=25, max_freqs=25, freq_times=1, time_times=1),
    SpecAugumentOptions(max_times=15, max_freqs=15, freq_times=2, time_times=2),
    SpecAugumentOptions(max_times=40, max_freqs=40, freq_times=1, time_times=1),
]

@dataclass
class BackgroundNoiseOptions:
    paths: List[str]
    probability: float

@dataclass
class AugumentationOptions:
    specaug_options: Optional[SpecAugumentOptions] = None
    bg_noise_options: Optional[BackgroundNoiseOptions] = None


def _apply_specaug(spectrogram: torch.Tensor, options: SpecAugumentOptions, rnd: Optional[np.random.RandomState]) -> torch.Tensor:
    if rnd is None:
        rnd = np.random.RandomState()
    spectrogram = spectrogram.detach().clone()

    for _ in range(options.freq_times):
        f = rnd.randint(low=1, high=options.max_freqs)
        f0 = rnd.randint(low=0, high=spectrogram.shape[1] - f)
        spectrogram[:, f0:f0+f] = 0.

    for _ in range(options.time_times):
        t = rnd.randint(low=1, high=options.max_times)
        t0 = rnd.randint(low=0, high=spectrogram.shape[0] - t)
        spectrogram[t0:t0+t, :] = 0.

    return spectrogram


class SelfPretrainDataset(torch.utils.data.IterableDataset):

    def __init__(self, paths: List[str], seed: int, xerox: Optional[int], augumentation_options: Optional[AugumentationOptions]):
        """
        - paths - paths to .wav files
        - seed - random seed
        - xerox - how many times to duplicates each sample. If None, generate infinite set.
        if not None, generate fixed set (for validation and test)
        """
        self._paths = paths
        self._rnd = np.random.RandomState(seed=seed)
        self._waveform_cache = {}
        self._fixed_set = None
        if augumentation_options is not None:
            self._augumentation_options = augumentation_options
        else:
            self._augumentation_options = AugumentationOptions()

        if xerox is not None:
            self._prepare_fixed_set(xerox)

    def _get_waveform(self, fname):
        if fname in self._waveform_cache:
            return self._waveform_cache[fname]

        waveform, sample_rate = torchaudio.load(fname)
        if sample_rate != SAMPLE_RATE:
            raise ValueError(f'{fname} has sample rate {sample_rate}, not {SAMPLE_RATE}')

        waveform = waveform[0].cpu().detach().numpy()
        self._waveform_cache[fname] = waveform
        return waveform

    def _get_features(self, waveform):
        waveform = torch.from_numpy(np.array([waveform]))
        return torchaudio.compliance.kaldi.fbank(
            waveform,
            num_mel_bins=80,
            frame_shift=10,
            frame_length=25
        )

    def _apply_bg_noise(self, waveform):
        if self._rnd.uniform() > self._augumentation_options.bg_noise_options.probability:
            return waveform
        return add_noise(
            waveform,
            self._read_audio(self._rnd.choice(self._augumentation_options.bg_noise_options.paths)),
            rnd
        )

    def _get_sample(self, fname):
        waveform = ensure_duration(self._get_waveform(fname))
        if self._augumentation_options.bg_noise_options is not None:
            waveform = self._apply_bg_noise(waveform)
        spectrogram = self._get_features(waveform)
        if self._augumentation_options.specaug_options is not None:
            spectrogram = _apply_specaug(spectrogram, self._augumentation_options.specaug_options, self._rnd)
        return spectrogram

    def _prepare_fixed_set(self, xerox):
        LOGGER.info('Preparing fixed set')
        self._fixed_set = []
        for i, path in enumerate(self._paths):
            for _ in range(xerox):
                self._fixed_set.append({
                    'data': self._get_sample(path),
                    'label': i
                })
            if (i + 1) % 100 == 0:
                LOGGER.debug('Prepared %d/%d samples for fixed set', i + 1, len(self._paths))
        LOGGER.info('Finished preparing fixed set')

    def __iter__(self):
        if self._fixed_set:
            yield from self._fixed_set
            return

        while True:
            label = self._rnd.randint(0, len(self._paths))
            spectrogram = self._get_sample(self._paths[label])
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
