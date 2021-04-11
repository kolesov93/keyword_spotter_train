#!/usr/bin/env python3.6
import os
import sys
import argparse
import subprocess as sp


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('results')
    parser.add_argument('wav_scp')
    parser.add_argument('data')
    return parser.parse_args()


def main():
    args = parse_args()
    if os.path.exists(args.data):
        raise ValueError(f'{args.data} already exists')

    u2path = {}
    with open(args.wav_scp) as fin:
        for line in fin:
            tokens = line.strip().split()
            assert len(tokens) == 2, line.strip()
            u2path[tokens[0]] = tokens[1]

    os.makedirs(args.data)

    with open(args.results) as fin:
        cword = None
        for line in fin:
            tokens = line.strip().split()
            assert len(tokens) in [1, 3], line.strip()
            if len(tokens) == 1:
                cword = tokens[0]
                if cword != '<sil>':
                    os.makedirs(os.path.join(args.data, cword))
                continue

            assert cword is not None, line.strip()
            if cword == '<sil>':
                continue
            uttid = tokens[0]
            shift = int(tokens[1])
            value = float(tokens[2])
            if value < 0.7:
                continue

            fname = os.path.join(args.data, cword, 'pseudo-{}-{}.wav'.format(uttid, shift))
            sp.check_call(
                ['sox', u2path[uttid], fname, 'trim', '{:.4f}'.format(shift / 16000.), '1']
            )


if __name__ == '__main__':
    main()
