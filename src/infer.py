#!/usr/bin/env python3
"""Train keyword spotter."""
import argparse
import copy
import faulthandler
import json
import logging
import os
import sys

from typing import Dict

import numpy as np
import torch
import torch.utils.data
import torch.nn as nn

from data.infer import InferenceDataset
import tabulate

import models.resnet as resnet
import models.wav2vec as model_wav2vec
import models.wav2vec_resnet as model_wav2vec_resnet
import models.fbank_ff as model_fbank_ff

LOGGER = logging.getLogger('spotter_infer')


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', help='path/to/final.mdl')
    parser.add_argument('options', help='path/to/options.json')
    parser.add_argument('wav', help='path/to/file.wav')
    return parser.parse_args()


def collate_fn(samples):
    x, sample_ids = [], []
    for sample in samples:
        x.append(sample['data'].reshape(-1))
        sample_ids.append((sample['uttid'], sample['shift']))

    x = np.array(x)
    return torch.from_numpy(x), sample_ids


def _get_model(args):
    with open(args.options) as fin:
        options = json.load(fin)

    arch = options['model']
    LOGGER.info('The model has arch %s', arch)

    if arch in ['ff', 'res8']:
        config = copy.deepcopy(resnet.RES8_CONFIG)
    elif arch == 'res8_narrow':
        config = copy.deepcopy(resnet.RES8_NARROW_CONFIG)
    elif arch == 'res15':
        config = copy.deepcopy(resnet.RES15_CONFIG)
    elif arch == 'res15_narrow':
        config = copy.deepcopy(resnet.RES15_NARROW_CONFIG)
    elif arch == 'res26':
        config = copy.deepcopy(resnet.RES26_CONFIG)
    elif arch == 'res26_narrow':
        config = copy.deepcopy(resnet.RES26_NARROW_CONFIG)

    config['n_labels'] = len(options['wanted_words']) + 2
    config['model_path'] = options.get('model_path')

    LOGGER.info('Getting the model with config "%s"', config)

    if options['use_fbank']:
        if arch == 'ff':
            model = model_fbank_ff.FbankFFModel(config)
        else:
            model = resnet.SpeechResModel(config)
    else:
        if arch == 'ff':
            model = model_wav2vec.FFModel(config)
        else:
            model = model_wav2vec_resnet.ResnetModel(config)
    return model


def infer(args):
    torch.set_num_threads(8)
    model = _get_model(args)
    model.load(args.model)
    model.eval()
    torch.cuda.set_device(0)
    model.cuda()

    u2path = {'sample': args.wav}
    dataset = InferenceDataset(u2path, 8000)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=32,
        collate_fn=collate_fn
    )

    result = {}
    softmax = nn.Softmax(dim=1).cuda()
    for x, sample_ids in loader:
        torch.cuda.set_device(0)
        x = x.cuda()
        scores = softmax(model(x)).detach().cpu().numpy()
        for sample_id, score in zip(sample_ids, scores):
            result[sample_id] = score

    return result


def main(args):
    """Run train."""
    LOGGER.setLevel(logging.INFO)
    LOGGER.info(sys.argv)

    try:
        result = infer(args)
    except Exception as e:
        LOGGER.exception(e)
        raise

    with open(args.options) as fin:
        options = json.load(fin)

    header = ['uttid', '<sil>', '<unk>'] + options['wanted_words']
    rows = []
    for (uttid, shift), scores in result.items():
        row = ['{}-{}'.format(uttid, shift)]
        for score in scores:
            row.append('{:.2f}'.format(score))
        rows.append(row)

    print(tabulate.tabulate(rows, headers=header))


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    faulthandler.enable()
    main(_parse_args())
