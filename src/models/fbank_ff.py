import torch.nn as nn
import torch
import torchaudio
from fairseq.models.wav2vec.wav2vec2 import Wav2Vec2Model


RES8_CONFIG = dict(n_labels=12, n_layers=6, n_feature_maps=45, res_pool=(4, 3), use_dilation=False)
import torch.nn.functional as F

class SerializableModule(nn.Module):
    def __init__(self):
        super().__init__()

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def load(self, filename):
        self.load_state_dict(torch.load(filename, map_location=lambda storage, loc: storage))

class FbankFFModel(SerializableModule):
    def __init__(self, config):
        super().__init__()
        n_labels = config["n_labels"]
        self._init_linear(n_labels)

    def _init_linear(self, n_labels):
        # batch x 98 x 80
        units = [128, 64]
        timestamps = 98
        prev = 80
        self._linears = []
        for i, unit in enumerate(units):
            self.add_module('linear{}'.format(i), torch.nn.Linear(prev, unit))
            prev = unit
        self.add_module('final_linear', torch.nn.Linear(prev * timestamps, n_labels))
        self._nlinears = len(units)


    def forward(self, x):
        fbanks = []
        for channel in x:
            fbanks.append(
                torchaudio.compliance.kaldi.fbank(
                    channel.reshape(1, -1),
                    num_mel_bins=80,
                    frame_shift=10,
                    frame_length=25
                )
            )
        x = torch.stack(fbanks)
        for i in range(self._nlinears):
            linear = getattr(self, 'linear{}'.format(i))
            x = linear(x)
            x = F.relu(x)
        x = x.reshape(x.shape[0], -1)
        x = self.final_linear(x)
        return x


