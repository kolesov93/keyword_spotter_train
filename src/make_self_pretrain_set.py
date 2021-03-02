#!/usr/bin/env python3.6
import argparse
import subprocess as sp
import numpy as np
import sys
import os


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('wav', nargs='+')
    parser.add_argument('n', type=int)
    parser.add_argument('data')
    return parser.parse_args()


def main():
    args = _parse_args()
    if os.path.exists(args.data):
        raise ValueError(f'{args.data} already exists')

    durations = []
    for wav in args.wav:
        durations.append(float(sp.check_output(['soxi', '-D', wav])))
    sum_durations = sum(durations)

    fname_len = len(str(args.n - 1))
    fmt = '{:0' + str(fname_len) + 'd}.wav'
    np.random.seed(1993)
    os.makedirs(args.data)
    with open(os.path.join(args.data, 'info'), 'w') as fout:
        for j in range(args.n):
            while True:
                start = np.random.uniform(0.0, sum_durations)
                chosen_wav = None
                for i, duration in enumerate(durations):
                    if start <= duration:
                        chosen_wav = args.wav[i]
                        break
                if chosen_wav is not None and start + 1.1 < duration:
                    break

            fname = os.path.join(args.data, fmt.format(j))
            print(f'{j}\t{fname}\t{chosen_wav}\t{start}', file=fout)
            sp.check_call(['sox', chosen_wav, fname, 'trim', '{:.4f}'.format(start), '1'])


if __name__ == '__main__':
    main()
