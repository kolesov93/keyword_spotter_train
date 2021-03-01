#!/usr/bin/env python3
"""Train keyword spotter."""
import argparse
import copy
import enum
import faulthandler
import json
import logging
import math
import os
import sys

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
import models.wav2vec as model_wav2vec
import models.wav2vec_resnet as model_wav2vec_resnet
import models.fbank_ff as model_fbank_ff

LOGGER = logging.getLogger('spotter_train')

DEV_PERCENTAGE = 10.
TEST_PERCENTAGE = 10.

WANTED_WORDS = 'yes,no,up,down,left,right,on,off,stop,go'

DUMP_SUMMARY_EVERY_STEPS = 20
MAX_PLATEAUS = 5

class Metrics(enum.Enum):
    ACCURACY = 'accuracy'
    XENT = 'xent'


MODEL_PATH = '/home/kolesov93/study/wav2vec_models/wav2vec_small.pt'

def make_wav2vec_collate_fn(args):
    # cp = torch.load(MODEL_PATH)
    # model = Wav2Vec2Model.build_model(cp['args'])
    # model.eval()

    def collate_fn(samples):
        x, y = [], []
        for sample in samples:
            x.append(sample['data'].reshape(-1))
            y.append(sample['label'])

        x = np.array(x)
        #features = model.feature_extractor(torch.tensor(np.array(x)))
        #x = features.permute(0, 2, 1)
        y = np.array(y, dtype=np.int64)
        return torch.from_numpy(x), torch.from_numpy(y)

    return collate_fn


def make_collate_fn(args):

    def collate_fn(samples):
        x, y = [], []
        for sample in samples:
            x.append(
                torchaudio.compliance.kaldi.fbank(
                    torch.from_numpy(sample['data'].reshape(1, -1)),
                    num_mel_bins=args.num_mel_bins,
                    frame_shift=args.frame_shift,
                    frame_length=args.frame_length,
                ).reshape(1, -1, args.num_mel_bins)
            )
            y.append(sample['label'])
        x = torch.cat(x, 0)
        y = np.array(y, dtype=np.int64)
        return x, torch.from_numpy(y)

    return collate_fn

def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--initialize-body', type=str, default=None, help='path/to/mdl to initialize body (without head) with')
    parser.add_argument('--limit', type=int, default=None, help='leave this number of samples per word')
    parser.add_argument('--batch-size', type=int, default=64, help='batch size')
    parser.add_argument('--lr', type=float, default=1e-1, help='starting learning rate')
    parser.add_argument('--frame-length', type=float, default=25., help='frame length in ms')
    parser.add_argument('--frame-shift', type=float, default=10., help='frame shift in ms')
    parser.add_argument('--num-mel-bins', type=int, default=80, help='num mel filters')
    parser.add_argument('--dev-every-batches', type=int, default=128, help='dev every batches')
    parser.add_argument('--lr-drop', type=float, default=1.5, help='decrease lr if platuea')
    parser.add_argument('--min-lr', type=float, default=1e-8, help='stop the training if lr is less than this value')
    parser.add_argument('--max-batches', type=int, default=60000, help='stop the training if the number of batches is bigger than this value')
    parser.add_argument('--model-path', default=None)
    parser.add_argument(
        '--model', required=True,
        choices='res8,res8_narrow,res15,res15_narrow,res26,res26_narrow,ff'.split(',')
    )
    parser.add_argument('--use-fbank', action='store_true')
    parser.add_argument('--wanted-words', default=WANTED_WORDS)
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


def evaluate(model, dataset, collate_fn):
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

def _initialize_logging(traindir):
    handler = logging.FileHandler(os.path.join(traindir, 'log'))
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    handler.setLevel(logging.DEBUG)
    LOGGER.addHandler(handler)


def _get_model(args):
    if args.model in ['ff', 'res8']:
        config = copy.deepcopy(resnet.RES8_CONFIG)
    elif args.model == 'res8_narrow':
        config = copy.deepcopy(resnet.RES8_NARROW_CONFIG)
    elif args.model == 'res15':
        config = copy.deepcopy(resnet.RES15_CONFIG)
    elif args.model == 'res15_narrow':
        config = copy.deepcopy(resnet.RES15_NARROW_CONFIG)
    elif args.model == 'res26':
        config = copy.deepcopy(resnet.RES26_CONFIG)
    elif args.model == 'res26_narrow':
        config = copy.deepcopy(resnet.RES26_NARROW_CONFIG)
    config['n_labels'] = len(args.wanted_words) + 2
    config['model_path'] = args.model_path
    if args.use_fbank:
        if args.model == 'ff':
            model = model_fbank_ff.FbankFFModel(config)
        else:
            model = resnet.SpeechResModel(config)
    else:
        if args.model == 'ff':
            model = model_wav2vec.FFModel(config)
        else:
            model = model_wav2vec_resnet.ResnetModel(config)
    return model


def train(args, sets):
    torch.set_num_threads(8)
    if os.path.exists(args.traindir):
        raise ValueError(f'{args.traindir} already exists')

    os.makedirs(args.traindir)

    summary_writer = SummaryWriter(os.path.join(args.traindir, 'summaries'))
    _initialize_logging(args.traindir)

    with open(os.path.join(args.traindir, 'options.json'), 'w') as fout:
        json.dump(vars(args), fout)

    collate_fn = make_wav2vec_collate_fn(args)

    model = _get_model(args)
    if args.initialize_body:
        LOGGER.info('Initializing body from %s', args.initialize_body)
        model.load(args.initialize_body, without_head=True)
        LOGGER.info('Done')

    prev_eval_metrics = evaluate(model, sets[DatasetTag.DEV], collate_fn)
    _dump_val_metrics(summary_writer, prev_eval_metrics, 0)
    prev_model_fname = os.path.join(args.traindir, 'model_0.mdl')
    model.save(prev_model_fname)
    with open(os.path.join(args.traindir, 'dev_metrics_0.json'), 'w') as fout:
        json.dump({m.value: value for m, value in prev_eval_metrics.items()}, fout)

    loader = torch.utils.data.DataLoader(
        sets[DatasetTag.TRAIN],
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        num_workers=4
    )

    criterion = nn.CrossEntropyLoss()
    curlr = args.lr
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

        force_stop = (curlr < args.min_lr) or (batch_idx + 1 > args.max_batches)
        if (batch_idx + 1) % args.dev_every_batches == 0 or force_stop:
            new_eval_metrics = evaluate(model, sets[DatasetTag.DEV], collate_fn)
            new_model_fname = os.path.join(args.traindir, 'model_{}.mdl'.format(batch_idx + 1))
            model.save(new_model_fname)
            _dump_val_metrics(summary_writer, new_eval_metrics, batch_idx + 1)
            with open(os.path.join(args.traindir, 'dev_metrics_{}.json'.format(batch_idx + 1)), 'w') as fout:
                json.dump({m.value: value for m, value in new_eval_metrics.items()}, fout)

            if new_eval_metrics[Metrics.ACCURACY] <= prev_eval_metrics[Metrics.ACCURACY] or math.isnan(new_eval_metrics[Metrics.XENT]):
                nplateaus += 1
                if nplateaus > MAX_PLATEAUS:
                    LOGGER.warning('Hitted plateau %d times, exiting', nplateaus)
                    break
                curlr = curlr / args.lr_drop
                optimizer = torch.optim.SGD(
                    model.parameters(),
                    lr=curlr,
                    momentum=0.9
                )
                LOGGER.warning(
                    'Accuracy is getting worse (%.02f vs %.02f) or xent is nan, decreasing lr to: %f',
                    new_eval_metrics[Metrics.ACCURACY],
                    prev_eval_metrics[Metrics.ACCURACY],
                    curlr
                )
                model.load(prev_model_fname)
                LOGGER.warning(
                    'Reloaded model from %s', prev_model_fname
                )
            else:
                if prev_model_fname != new_model_fname:
                    os.unlink(prev_model_fname)
                prev_eval_metrics = new_eval_metrics
                prev_model_fname = new_model_fname

            if force_stop:
                LOGGER.warning('Training is force stopped. LR: %f, batches: %d', curlr, batch_idx + 1)
                break

    LOGGER.info('Best model is in %s', prev_model_fname)
    model.load(prev_model_fname)
    test_metrics = evaluate(model, sets[DatasetTag.TEST], collate_fn)
    with open(os.path.join(args.traindir, 'test_metrics.json'), 'w') as fout:
        json.dump({m.value: value for m, value in test_metrics.items()}, fout)

    for fname in os.listdir(args.traindir):
        if fname.endswith('.mdl'):
            os.unlink(os.path.join(args.traindir, fname))

    model.save(os.path.join(args.traindir, 'final.mdl'))


def main(args):
    """Run train."""
    LOGGER.setLevel(logging.INFO)
    LOGGER.info(sys.argv)
    args.wanted_words = args.wanted_words.split(',')

    indexes = gsc.split_index(gsc.get_index(args.data), DEV_PERCENTAGE, TEST_PERCENTAGE, args.limit)
    sets = [
        gsc.GoogleSpeechCommandsDataset(
            indexes[tag],
            args.wanted_words,
            tag
        )
        for tag in DatasetTag
    ]

    try:
        train(args, sets)
    except Exception as e:
        LOGGER.exception(e)
        raise

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    faulthandler.enable()
    main(_parse_args())
