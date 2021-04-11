#!/usr/bin/env python3
import argparse
import sys
import numpy as np

def read_probs(fname):
    result = {}  # uttid -> {word -> prob}

    words = []
    with open(fname) as fin:
        for i, line in enumerate(fin):
            tokens = line.strip().split()
            if i == 0:
                for token in tokens:
                    if token not in ('winner', 'uttid', 'shift'):
                        words.append(token)
                continue
            if i == 1:
                continue

            uttid = tokens[0]
            result[uttid] = {
                word: float(tokens[2 + j])
                for j, word in enumerate(words)
            }
    return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('fname', nargs='+')
    return parser.parse_args()


def convert(probs):
    uttids = list(probs[0].keys())
    words = list(probs[0][uttids[0]].keys())

    result = np.zeros((len(probs), len(uttids), len(words)), dtype=np.float32)
    for i, prob in enumerate(probs):
        for j, uttid in enumerate(uttids):
            for k, word in enumerate(words):
                result[i][j][k] = probs[i][uttid][word]
    return result, uttids, words


def compute_accuracy(prob_matrix, uttids, words):
    num, denom = 0, 0
    for i, probs in enumerate(prob_matrix):
        target = uttids[i].split('-')[0]
        if target not in words:
            target = '<unk>'

        answer = words[np.argmax(probs)]
        if answer == target:
            num += 1
        denom += 1
    return num / denom


def main():
    args = parse_args()
    probs = []
    for fname in args.fname:
        probs.append(read_probs(fname))
    probs, uttids, words = convert(probs)

    if len(probs) == 1:
        print(compute_accuracy(probs[0], uttids, words) * 100)
        return

    if len(probs) == 2:
        best_w = None
        best_acc = None
        for w in np.arange(0., 1.0, 0.01):
            ws = np.array([w, 1. - w]).reshape([2, 1, 1])
            sumprobs = np.sum(probs * ws, axis=0)
            acc = compute_accuracy(sumprobs, uttids, words)
            if best_acc is None or acc > best_acc:
                best_acc = acc
                best_w = w
        print('Best w: {:.02f}'.format(best_w), file=sys.stderr)
        print(best_acc * 100)
        return

    ITER = 10000
    for _ in range(ITER):
        best_ws = None
        best_acc = None
        ws = np.random.uniform(size=(len(probs), 1, 1))
        sumprobs = np.sum(probs * ws, axis=0)
        acc = compute_accuracy(sumprobs, uttids, words)
        if best_acc is None or acc > best_acc:
            best_acc = acc
            best_ws = ws
    print('Best ws: {}'.format(best_ws), file=sys.stderr)
    print(best_acc * 100)


if __name__ == '__main__':
    main()
