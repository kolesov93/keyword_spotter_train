import logging
import enum

import torch.utils.data
import torch
import torch.nn as nn


LOGGER = logging.getLogger('spotter_training')


class Metrics(enum.Enum):
    ACCURACY = 'accuracy'
    XENT = 'xent'


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
