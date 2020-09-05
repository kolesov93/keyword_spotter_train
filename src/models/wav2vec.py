import torch.nn as nn
import torch
import torchaudio
from fairseq.models.wav2vec import Wav2VecModel

RES8_CONFIG = dict(n_labels=12, n_layers=6, n_feature_maps=45, res_pool=(4, 3), use_dilation=False)
import torch.nn.functional as F

class SerializableModule(nn.Module):
    def __init__(self):
        super().__init__()

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def load(self, filename):
        self.load_state_dict(torch.load(filename, map_location=lambda storage, loc: storage))

class FFModel(SerializableModule):
    def __init__(self, config):
        super().__init__()
        n_labels = config["n_labels"]
        self._init_feature_extractor(config['model_path'])
        self._init_linear(n_labels)

    def _init_feature_extractor(self, model_path):
        cp = torch.load(model_path)
        model = Wav2VecModel.build_model(cp['args'], task=None)
        model.load_state_dict(cp['model'])
        model.eval()
        self._feature_extractor = model.feature_extractor
        self._feature_aggregator = model.feature_aggregator

    def _init_linear(self, n_labels):
        # batch x 98 x 512
        units = [128, 64]
        timestamps = 98
        prev = 512
        self._linears = []
        for i, unit in enumerate(units):
            self.add_module('linear{}'.format(i), torch.nn.Linear(prev, unit))
            prev = unit
        self.add_module('final_linear', torch.nn.Linear(prev * timestamps, n_labels))
        self._nlinears = len(units)

    def forward(self, x):
        x = self._feature_extractor(x)
        x = self._feature_aggregator(x)
        x = x.permute(0, 2, 1)
        for i in range(self._nlinears):
            linear = getattr(self, 'linear{}'.format(i))
            x = linear(x)
            x = F.relu(x)
        x = x.reshape(x.shape[0], -1)
        x = self.final_linear(x)
        return x


