#!/usr/bin/env python3
import argparse
import os
import json
import sys

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sort', default='accuracy')
    parser.add_argument('traindirs')
    return parser.parse_args()

def main():
    args = parse_args()
    rows = []
    for d in os.listdir(args.traindirs):
        d = os.path.join(args.traindirs, d)
        if not os.path.isdir(d):
            continue

        opt_file = os.path.join(d, 'options.json')
        metrics_file = os.path.join(d, 'test_metrics.json')
        if not (os.path.exists(opt_file) and os.path.exists(metrics_file)):
            print('Skipping {}, some files absent'.format(d), file=sys.stderr)
            continue

        with open(opt_file) as fin:
            options = json.load(fin)
        with open(metrics_file) as fin:
            metrics = json.load(fin)

        options.pop('traindir')
        options.pop('data')
        result = options
        result.update(metrics)
        rows.append(result)

    keys = list(rows[0].keys())
    keys.remove(args.sort)
    keys.append(args.sort)
    print('\t'.join(keys))

    rows.sort(key=lambda x: x[args.sort])
    for row in rows:
        values = [str(row[key]) for key in keys]
        print('\t'.join(values))


if __name__ == '__main__':
    main()

