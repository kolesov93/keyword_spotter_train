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
import data.self_pretrain as selfp

import models.resnet as resnet
import models.wav2vec as model_wav2vec
import models.wav2vec_resnet as model_wav2vec_resnet
import models.fbank_ff as model_fbank_ff

LOGGER = logging.getLogger('spotter_train')
LOGGER_FORMAT = '%(asctime)s - %(pathname)s:%(lineno)d - %(levelname)s - %(message)s'

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
    def collate_fn(samples):
        x, y = [], []
        for sample in samples:
            x.append(sample['data'].reshape(-1))
            y.append(sample['label'])

        x = np.array(x)
        y = np.array(y, dtype=np.int64)
        return torch.from_numpy(x), torch.from_numpy(y)

    return collate_fn


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--initialize-body', type=str, default=None, help='path/to/mdl to initialize body (without head) with')
    parser.add_argument('--initialize-all', type=str, default=None, help='path/to/mdl to initialize all model with')
    parser.add_argument('--limit', type=int, default=None, help='leave this number of samples per word')
    parser.add_argument('--specaug-level', type=int, default=1)
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
    parser.add_argument('--self-pretrain', action='store_true')
    parser.add_argument('data', help='path/to/google_speech_command/dataset')
    parser.add_argument('traindir', help='path/to/traindir')
    return parser.parse_args()


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
    formatter = logging.Formatter(LOGGER_FORMAT)
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

    if args.self_pretrain:
        config['from_fbank'] = True
        config['n_labels'] = len(list(fname for fname in os.listdir(args.data) if fname.endswith('.wav')))

    else:
        config['n_labels'] = len(args.wanted_words) + 2
    config['model_path'] = args.model_path

    LOGGER.info('Using config %s', config)
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


def _are_new_metrics_worse(prev_metrics, new_metrics):
    if math.isnan(new_metrics[Metrics.XENT]):
        LOGGER.warning('New xent is nan')
        return True
    if prev_metrics[Metrics.ACCURACY] > new_metrics[Metrics.ACCURACY]:
        LOGGER.warning(
            'New accuracy is worse, than previous best: %.02f vs %.02f',
            new_metrics[Metrics.ACCURACY],
            prev_metrics[Metrics.ACCURACY]
        )
        return True
    if prev_metrics[Metrics.ACCURACY] == new_metrics[Metrics.ACCURACY]:
        if prev_metrics[Metrics.XENT] <= new_metrics[Metrics.XENT]:
            LOGGER.warning(
                'Accuracy hasn\'t changed (%.02f), but new xent is not better, than previous best: %.02f vs %.02f',
                new_metrics[Metrics.ACCURACY],
                new_metrics[Metrics.XENT],
                prev_metrics[Metrics.XENT]
            )
            return True
        LOGGER.info(
            'Accuracy hasn\'t changed (%.02f), but new xent is better, than previous best: %.02f vs %.02f',
                new_metrics[Metrics.ACCURACY],
                new_metrics[Metrics.XENT],
                prev_metrics[Metrics.XENT]
        )
        return False

    LOGGER.info(
        'New accuracy is better, than previous best: %.02f vs %.02f',
        new_metrics[Metrics.ACCURACY],
        prev_metrics[Metrics.ACCURACY]
    )
    if prev_metrics[Metrics.XENT] < new_metrics[Metrics.XENT]:
        LOGGER.warning(
            "New accuracy is better, but the new xent is worse: %.02f vs %.02f",
            new_metrics[Metrics.XENT],
            prev_metrics[Metrics.XENT]
        )
    return False


def train(args, sets):
    if not os.path.exists(args.traindir):
        raise ValueError(f'{args.traindir} doesn\'t exist')

    torch.set_num_threads(8)

    summary_writer = SummaryWriter(os.path.join(args.traindir, 'summaries'))

    with open(os.path.join(args.traindir, 'options.json'), 'w') as fout:
        json.dump(vars(args), fout)

    if args.self_pretrain:
        collate_fn = selfp.collate_fn
    else:
        collate_fn = make_wav2vec_collate_fn(args)

    model = _get_model(args)
    if args.initialize_body:
        assert not args.initialize_all
        LOGGER.info('Initializing body from %s', args.initialize_body)
        model.load(args.initialize_body, without_head=True)
        LOGGER.info('Done')
    elif args.initialize_all:
        LOGGER.info('Initializing whole model from %s', args.initialize_body)
        model.load(args.initialize_all)
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
            the_xent = loss.item()
            the_accuracy = compute_accuracy(scores, y)

            summary_writer.add_scalar(
                'train/xent',
                the_xent,
                batch_idx + 1
            )
            summary_writer.add_scalar(
                'train/accuracy',
                the_accuracy,
                batch_idx + 1
            )
            summary_writer.add_scalar(
                'train/lr',
                curlr,
                batch_idx + 1
            )

            LOGGER.info('batch %d, train xent: %.02f, train acc: %.01f%%', batch_idx + 1, the_xent, the_accuracy * 100.)

        force_stop = (curlr < args.min_lr) or (batch_idx + 1 > args.max_batches)
        if (batch_idx + 1) % args.dev_every_batches == 0 or force_stop:
            new_eval_metrics = evaluate(model, sets[DatasetTag.DEV], collate_fn)
            new_model_fname = os.path.join(args.traindir, 'model_{}.mdl'.format(batch_idx + 1))
            model.save(new_model_fname)
            _dump_val_metrics(summary_writer, new_eval_metrics, batch_idx + 1)
            with open(os.path.join(args.traindir, 'dev_metrics_{}.json'.format(batch_idx + 1)), 'w') as fout:
                json.dump({m.value: value for m, value in new_eval_metrics.items()}, fout)

            if _are_new_metrics_worse(prev_eval_metrics, new_eval_metrics):
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
                LOGGER.warning('Decreasing lr to: %f', curlr)
                model.load(prev_model_fname)
                LOGGER.warning('Reloaded model from %s', prev_model_fname)
            else:
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
    if os.path.exists(args.traindir):
        raise ValueError(f'{args.traindir} already exists')
    os.makedirs(args.traindir)

    _initialize_logging(args.traindir)

    LOGGER.info(sys.argv)

    args.wanted_words = args.wanted_words.split(',')

    if args.self_pretrain:
        paths = list(sorted([
            os.path.join(args.data, fname)
            for fname in os.listdir(args.data)
            if fname.endswith('.wav')
        ]))

        augumentation_options = selfp.AugumentationOptions(
            specaug_options=selfp.SPECAUG_LEVELS[args.specaug_level - 1],
            bg_noise_options=None
        )

        sets = {
            DatasetTag.TRAIN: selfp.SelfPretrainDataset(paths, 1993, None, augumentation_options),
            DatasetTag.DEV: selfp.SelfPretrainDataset(paths, 1994, 3, augumentation_options),
            DatasetTag.TEST: selfp.SelfPretrainDataset(paths, 1995, 1, None),
        }
        sets = [
            sets[key]
            for key in sorted(sets.keys())
        ]
    else:
        indexes = gsc.split_index(gsc.get_index(args.data), DEV_PERCENTAGE, TEST_PERCENTAGE, args.limit)
        for tag in DatasetTag:
            with open(os.path.join(args.traindir, '{}.files'.format(tag)), 'w') as fout:
                for label in indexes[tag]:
                    for fname in indexes[tag][label]:
                        print('{}\t{}'.format(fname, label), file=fout)
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
    logging.basicConfig(format=LOGGER_FORMAT, level=logging.DEBUG)
    faulthandler.enable()
    main(_parse_args())
