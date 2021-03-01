import torch.nn as nn
import torch
import torchaudio
from fairseq.models.wav2vec.wav2vec2 import Wav2Vec2Model


RES8_CONFIG = dict(n_labels=12, n_layers=6, n_feature_maps=45, res_pool=(4, 3), use_dilation=False)
RES8_NARROW_CONFIG = dict(n_labels=12, n_layers=6, n_feature_maps=19, res_pool=(4, 3), use_dilation=False)
RES15_CONFIG = dict(n_labels=12, use_dilation=True, n_layers=13, n_feature_maps=45)
RES15_NARROW_CONFIG = dict(n_labels=12, use_dilation=True, n_layers=13, n_feature_maps=19)
RES26_CONFIG = dict(n_labels=12, n_layers=24, n_feature_maps=45, res_pool=(2, 2), use_dilation=False)
RES26_NARROW_CONFIG = dict(n_labels=12, n_layers=24, n_feature_maps=19, res_pool=(2, 2), use_dilation=False)

import torch.nn.functional as F

class SerializableModule(nn.Module):
    def __init__(self):
        super().__init__()

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def load(self, filename, without_head=False):
        state_dict = torch.load(filename, map_location=lambda storage, loc: storage)
        if without_head:
            for key in list(state_dict.keys()):
                if key.startswith('output.'):
                    state_dict.pop(key)
        self.load_state_dict(state_dict, strict=not without_head)

class SpeechResModel(SerializableModule):
    def __init__(self, config):
        super().__init__()
        n_labels = config["n_labels"]
        n_maps = config["n_feature_maps"]
        self.conv0 = nn.Conv2d(1, n_maps, (3, 3), padding=(1, 1), bias=False)
        self.fbank_conv0 = nn.Conv2d(1, n_maps, (3, 3), padding=(1, 1), bias=False)
        if "res_pool" in config:
            self.pool = nn.AvgPool2d(config["res_pool"])
            self.fbank_pool = nn.AvgPool2d(config["res_pool"])

        self.n_layers = n_layers = config["n_layers"]
        dilation = config["use_dilation"]
        if dilation:
            self.convs = [nn.Conv2d(n_maps, n_maps, (3, 3), padding=int(2**(i // 3)), dilation=int(2**(i // 3)),
                bias=False) for i in range(n_layers)]
        else:
            self.convs = [nn.Conv2d(n_maps, n_maps, (3, 3), padding=1, dilation=1,
                bias=False) for _ in range(n_layers)]
        for i, conv in enumerate(self.convs):
            self.add_module("bn{}".format(i + 1), nn.BatchNorm2d(n_maps, affine=False))
            self.add_module("conv{}".format(i + 1), conv)

        self.output = nn.Linear(n_maps, n_labels)

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

        x = x.unsqueeze(1)
        for i in range(self.n_layers + 1):
            y = F.relu(getattr(self, "conv{}".format(i))(x))
            if i == 0:
                if hasattr(self, "pool"):
                    y = self.pool(y)
                old_x = y
            if i > 0 and i % 2 == 0:
                x = y + old_x
                old_x = x
            else:
                x = y
            if i > 0:
                x = getattr(self, "bn{}".format(i))(x)
        x = x.view(x.size(0), x.size(1), -1) # shape: (batch, feats, o3)
        return self.output(torch.mean(x, 2))
