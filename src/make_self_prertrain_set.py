#!/usr/bin/env python3.6
import argparse
import subprocess as sp
import numpy as np
import sys
import os


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('wav')
    parser.add_argument('n', type=int)
    parser.add_argument('data')
    return parser.parse_args()


def main():
    args = _parse_args()
    if os.path.exists(args.data):
        raise ValueError(f'{args.data} already exists')

    duration = float(sp.check_output(['soxi', '-D', args.wav]))
    print('Duration is {:.1f} seconds'.format(duration), file=sys.stderr)

    fname_len = len(str(args.n - 1))
    fmt = '{:0' + str(fname_len) + 'd}.wav'
    np.random.seed(1993)
    os.makedirs(args.data)
    with open(os.path.join(args.data, 'info'), 'w') as fout:
        for i in range(args.n):
            start = np.random.uniform(0.0, duration - 1.1)
            fname = os.path.join(args.data, fmt.format(i))
            print(f'{i}\t{fname}\t{start}', file=fout)
            sp.check_call(['sox', args.wav, fname, 'trim', '{:.4f}'.format(start), '1'])


if __name__ == '__main__':
    main()
