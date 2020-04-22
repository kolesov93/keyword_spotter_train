#!/usr/bin/env python3.8
"""Train keyword spotter."""
import argparse
import logging
import copy

import numpy as np
import torch
import torch.utils.data
import torch.nn as nn
import torchaudio
from torch.autograd import Variable
import torch.nn.functional as F

import data.google_speech_commands as gsc
from data.common import DatasetTag

import models.resnet as resnet

LOGGER = logging.getLogger('spotter_train')

DEV_PERCENTAGE = 10.
TEST_PERCENTAGE = 10.

WANTED_WORDS = 'yes,no,up,down,left,right,on,off,stop,go'.split(',')

BATCH_SIZE = 32
LR = 1e-2
DEV_EVERY_BATCHES = 1024

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


def evaluate(model, dataset):
    LOGGER.info('Starting evaluation')

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=32,
        collate_fn=collate_fn
    )
    model.eval()
    loss_sum = 0.
    nbatches = 0
    criterion = nn.CrossEntropyLoss()
    for i, (x, y) in enumerate(loader):
        scores = model(x)
        loss = criterion(scores, y)
        loss_sum += loss.item()
        nbatches += 1

        if nbatches % 10 == 0:
            LOGGER.info('Evaluation-xent [batch: %d]: %.02f', nbatches, loss_sum / nbatches)

    result_loss = loss_sum / nbatches
    LOGGER.info('Evaluation-xent: %.02f', result_loss)
    return result_loss

def train(args, sets):
    #model = FeedforwardModel([128, 128, 64], len(WANTED_WORDS) + 2)
    #model = LinearModel(16000, len(WANTED_WORDS) + 2)
    config = copy.deepcopy(resnet.RES8_CONFIG)
    config['n_labels'] = len(WANTED_WORDS) + 2
    model = resnet.SpeechResModel(config)

    prev_eval_loss = evaluate(model, sets[DatasetTag.DEV])

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
    )

    for batch_idx, (x, y) in enumerate(loader):
        model.train()
        optimizer.zero_grad()
        x = Variable(x, requires_grad=False)
        scores = model(x)
        y = Variable(y, requires_grad=False)
        loss = criterion(scores, y)
        loss.backward()
        optimizer.step()

        if batch_idx % 10 == 0:
            LOGGER.info('Train-xent [batch %d]: %.02f', batch_idx, loss.item())

        if (batch_idx + 1) % DEV_EVERY_BATCHES == 0:
            new_eval_loss = evaluate(model, sets[DatasetTag.DEV])
            if new_eval_loss > prev_eval_loss:
                LOGGER.info('Evaluation loss (%.02f) is bigger than previous (%.02f)', new_eval_loss, prev_eval_loss)
                curlr = curlr / 1.5
                optimizer = torch.optim.SGD(
                    model.parameters(),
                    lr=curlr,
                )
            else:
                LOGGER.info('Evaluation loss (%.02f) is less than previous (%.02f); continue', new_eval_loss, prev_eval_loss)
                prev_eval_loss = new_eval_loss


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
