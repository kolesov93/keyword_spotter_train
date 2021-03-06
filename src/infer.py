#!/usr/bin/env python3.6
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
    parser.add_argument('--sample-shift', type=int, default=8000, help='shift in samples (default: %(default)d)')
    parser.add_argument('--treat-as-wav-scp', action='store_true')
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

    if args.treat_as_wav_scp:
        u2path = {}
        with open(args.wav) as fin:
            for line in fin:
                tokens = line.strip().split()
                assert len(tokens) == 2, line.rstrip()
                u2path[tokens[0]] = tokens[1]
    else:
        u2path = {'sample': args.wav}

    dataset = InferenceDataset(u2path, args.sample_shift)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=128,
        collate_fn=collate_fn
    )

    result = {}
    softmax = nn.Softmax(dim=1).cuda()
    done = 0
    for x, sample_ids in loader:
        torch.cuda.set_device(0)
        x = x.cuda()
        scores = softmax(model(x)).detach().cpu().numpy()
        for sample_id, score in zip(sample_ids, scores):
            result[sample_id] = score
        done += len(sample_ids)
        LOGGER.info('Done %d samples', done)

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

    outputs = ['<sil>', '<unk>'] + options['wanted_words']
    header = ['uttid', 'shift'] + outputs + ['winner']
    rows = []
    for (uttid, shift), scores in result.items():
        row = [uttid, str(shift)]
        for score in scores:
            row.append('{:.5f}'.format(score))
        row.append(outputs[np.argmax(scores)])
        rows.append(row)

    print(tabulate.tabulate(rows, headers=header))


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    faulthandler.enable()
    main(_parse_args())
