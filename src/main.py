#!/usr/bin/env python3
"""Train keyword spotter."""
import argparse
import copy
import enum
import json
import logging
import os

from typing import Dict

import numpy as np
import torch
import torch.utils.data
import torch.nn as nn
import torchaudio
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import data.google_speech_commands as gsc
from data.common import DatasetTag

import models.resnet as resnet

LOGGER = logging.getLogger('spotter_train')

DEV_PERCENTAGE = 10.
TEST_PERCENTAGE = 10.

WANTED_WORDS = 'yes,no,up,down,left,right,on,off,stop,go'.split(',')

BATCH_SIZE = 64
LR = 1e-1
DEV_EVERY_BATCHES = 1024
DUMP_SUMMARY_EVERY_STEPS = 20
MAX_PLATEAUS = 2

class Metrics(enum.Enum):
    ACCURACY = 'accuracy'
    XENT = 'xent'


def collate_fn(samples):
    x, y = [], []
    for sample in samples:
        x.append(
            torchaudio.compliance.kaldi.fbank(
                torch.from_numpy(sample['data'].reshape(1, -1)),
                num_mel_bins=80,
                dither=0.
            ).reshape(1, -1, 80)
        )
        y.append(sample['label'])
    x = torch.cat(x, 0)
    y = np.array(y, dtype=np.int64)
    return x, torch.from_numpy(y)

def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('data', help='path/to/google_speech_command/dataset')
    parser.add_argument('traindir', help='path/to/traindir')
    return parser.parse_args()

class LinearModel(torch.nn.Module):

    def __init__(self, D_in, D_out):
        super(LinearModel, self).__init__()
        self.linear = torch.nn.Linear(D_in, D_out)

    def forward(self, x):
        y = self.linear(x)
        return y

class FeedforwardModel(torch.nn.Module):

    def __init__(self, units, noutputs):
        super(FeedforwardModel, self).__init__()
        timestamps = 98
        prev = 80
        self._linears = []
        for unit in units:
            self._linears.append(torch.nn.Linear(prev, unit))
            prev = unit
        self._final = torch.nn.Linear(prev * timestamps, noutputs)

    def forward(self, x):
        batch_size = x.shape[0]

        xs = []
        for curx in x:
            xs.append(
                torchaudio.compliance.kaldi.fbank(curx.reshape(1, -1), num_mel_bins=80, dither=0.)
            )
        x = torch.cat(xs, 0).reshape(batch_size, -1, 80)

        for linear in self._linears:
            x = linear(x)
            x = F.relu(x)

        x = x.reshape(x.shape[0], -1)
        x = self._final(x)
        return x

class AccuracyComputer:
    def __init__(self):
        self._correct_predictions = 0
        self._all_predictions = 0

    def update(self, scores, y):
        _, predicted = torch.max(scores, 1)
        self._correct_predictions += (predicted == y).sum().item()
        self._all_predictions += y.size(0)

    def compute(self):
        return self._correct_predictions / self._all_predictions


class AverageLossComputer:
    def __init__(self):
        self._loss_sum = 0.
        self._times = 0

    def update(self, loss):
        self._loss_sum += loss
        self._times += 1

    def compute(self):
        return self._loss_sum / self._times


def compute_accuracy(scores, y):
    a = AccuracyComputer()
    a.update(scores, y)
    return a.compute()


def evaluate(model, dataset):
    LOGGER.info('Starting evaluation')

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=32,
        collate_fn=collate_fn
    )
    model.eval()
    torch.cuda.set_device(0)
    model.cuda()
    criterion = nn.CrossEntropyLoss().cuda()

    accuracy_computer = AccuracyComputer()
    loss_computer = AverageLossComputer()
    for i, (x, y) in enumerate(loader):
        torch.cuda.set_device(0)
        x, y = x.cuda(), y.cuda()
        scores = model(x)
        loss = criterion(scores, y)

        accuracy_computer.update(scores, y)
        loss_computer.update(loss.item())

    result = {
        Metrics.ACCURACY: accuracy_computer.compute() * 100.,
        Metrics.XENT: loss_computer.compute()
    }
    LOGGER.info('Evaluation is finished')
    for metric, value in result.items():
        LOGGER.info('%s: %.02f', metric, value)
    return result

def _dump_val_metrics(writer: SummaryWriter, metrics: Dict[Metrics, float], step: int):
    for metric, value in metrics.items():
        writer.add_scalar(
            'eval/{}'.format(metric),
            value,
            step
        )
    writer.flush()

def train(args, sets):
    if os.path.exists(args.traindir):
        raise ValueError(f'{args.traindir} already exists')

    os.makedirs(args.traindir)
    logdir = os.path.join(args.traindir, 'logs')
    summary_writer = SummaryWriter(logdir)

    #model = FeedforwardModel([128, 128, 64], len(WANTED_WORDS) + 2)
    #model = LinearModel(16000, len(WANTED_WORDS) + 2)
    config = copy.deepcopy(resnet.RES8_CONFIG)
    config['n_labels'] = len(WANTED_WORDS) + 2
    model = resnet.SpeechResModel(config)

    prev_eval_metrics = evaluate(model, sets[DatasetTag.DEV])
    _dump_val_metrics(summary_writer, prev_eval_metrics, 0)
    prev_model_fname = os.path.join(args.traindir, 'model_0.mdl')
    model.save(prev_model_fname)
    with open(os.path.join(args.traindir, 'dev_metrics_0.json'), 'w') as fout:
        json.dump({m.value: value for m, value in prev_eval_metrics.items()}, fout)

    loader = torch.utils.data.DataLoader(
        sets[DatasetTag.TRAIN],
        batch_size=BATCH_SIZE,
        collate_fn=collate_fn,
        num_workers=4
    )

    criterion = nn.CrossEntropyLoss()
    curlr = LR
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=curlr,
        momentum=0.9
    )

    nplateaus = 0
    model.cuda()
    for batch_idx, (x, y) in enumerate(loader):
        model.train()
        optimizer.zero_grad()

        x = Variable(x, requires_grad=False).cuda()
        y = Variable(y, requires_grad=False).cuda()

        scores = model(x)
        loss = criterion(scores, y)

        loss.backward()
        optimizer.step()

        if (batch_idx + 1) % DUMP_SUMMARY_EVERY_STEPS == 0:
            summary_writer.add_scalar(
                'train/xent',
                loss.item(),
                batch_idx + 1
            )
            summary_writer.add_scalar(
                'train/accuracy',
                compute_accuracy(scores, y),
                batch_idx + 1
            )
            summary_writer.add_scalar(
                'train/lr',
                curlr,
                batch_idx + 1
            )

        if (batch_idx + 1) % DEV_EVERY_BATCHES == 0:
            new_eval_metrics = evaluate(model, sets[DatasetTag.DEV])
            new_model_fname = os.path.join(args.traindir, 'model_{}.mdl'.format(batch_idx + 1))
            model.save(new_model_fname)
            _dump_val_metrics(summary_writer, new_eval_metrics, batch_idx + 1)
            with open(os.path.join(args.traindir, 'dev_metrics_{}.json'.format(batch_idx + 1)), 'w') as fout:
                json.dump({m.value: value for m, value in new_eval_metrics.items()}, fout)

            if new_eval_metrics[Metrics.ACCURACY] < prev_eval_metrics[Metrics.ACCURACY]:
                nplateaus += 1
                if nplateaus > MAX_PLATEAUS:
                    LOGGER.warning('Hitted plateau %d times, exiting', nplateaus)
                    break
                curlr = curlr / 1.5
                optimizer = torch.optim.SGD(
                    model.parameters(),
                    lr=curlr,
                    momentum=0.9
                )
                LOGGER.warning(
                    'Accuracy is getting worse (%.02f vs %.02f), decreasing lr to: %f',
                    new_eval_metrics[Metrics.ACCURACY],
                    prev_eval_metrics[Metrics.ACCURACY],
                    curlr
                )
                model.load(prev_model_fname)
                LOGGER.warning(
                    'Reloaded model from %s', prev_model_fname
                )
            else:
                prev_eval_metrics = new_eval_metrics
                prev_model_fname = new_model_fname

    LOGGER.info('Best model is in %s', prev_model_fname)
    model.load(prev_model_fname)
    test_metrics = evaluate(model, sets[DatasetTag.TEST])
    with open(os.path.join(args.traindir, 'test_metrics.json'), 'w') as fout:
        json.dump({m.value: value for m, value in test_metrics.items()}, fout)


def main(args):
    """Run train."""
    LOGGER.setLevel(logging.INFO)

    indexes = gsc.split_index(gsc.get_index(args.data), DEV_PERCENTAGE, TEST_PERCENTAGE)
    sets = [
        gsc.GoogleSpeechCommandsDataset(
            indexes[tag],
            WANTED_WORDS,
            tag
        )
        for tag in DatasetTag
    ]

    train(args, sets)

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    main(_parse_args())
