#!/usr/bin/env python3
import argparse
import os
import json
import sys

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sort', default='accuracy')
    parser.add_argument('traindirs', nargs='+')
    return parser.parse_args()

def main():
    args = parse_args()
    rows = []
    for traindirs in args.traindirs:
        for d in os.listdir(traindirs):
            d = os.path.join(traindirs, d)
            if not os.path.isdir(d):
                continue

            opt_file = os.path.join(d, 'options.json')
            metrics_file = os.path.join(d, 'test_metrics.json')
            skip = False
            for path in [opt_file, metrics_file]:
                if not os.path.exists(path):
                    print('Skipping {}, {} is absent'.format(d, path), file=sys.stderr)
                    skip = True
                    break
            if skip:
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

