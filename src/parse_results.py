#!/usr/bin/env python3
import argparse
import json
import sys
import os

NEEDED = ['limit', 'use_fbank', 'batch_size', 'dev_every_batches', 'model', 'lr', 'lr_drop', 'data']


def _get_fnames():
    result = []
    for line in sys.stdin:
        result.append(line.strip())
    return result

def _get_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--header', action='store_true')
    parser.add_argument('--group-by', nargs='+')
    return parser.parse_args()


def _get_key(options, keys):
    return tuple(options[key] or -1 for key in keys)

def _get_lang(data):
    if 'rus_data' in data:
        return 'ru'
    if 'lt_data' in data:
        return 'lt'
    return 'en'


def _get_features(use_fbank):
    if use_fbank:
        return 'fbank'
    return 'wav2vec'


def _get_model(model):
    return '\\texttt{' + model.replace('_', '-') + '}'


def main():
    fnames = _get_fnames()
    args = _get_params()

    results = []
    for fname in fnames:
        with open(fname) as fin:
            metrics = json.load(fin)
        with open(os.path.join(os.path.dirname(fname), 'options.json')) as fin:
            options = json.load(fin)
        if any(field not in options for field in NEEDED):
            continue
        results.append((options, metrics))
    print('Loaded {} entries'.format(len(results)), file=sys.stderr)

    key2best = {}
    for options, metrics in results:
        key = _get_key(options, args.group_by)
        if key in key2best:
            if metrics['accuracy'] > key2best[key][1]['accuracy']:
                key2best[key] = (options, metrics)
        else:
            key2best[key] = (options, metrics)


    if args.header:
        row = NEEDED + ['accuracy', 'xent']
        print(' & '.join(row) + ' \\\\')

    for key in sorted(key2best):
        options, metrics = key2best[key]
        row = []
        for field in NEEDED:
            if field == 'data':
                row.append(_get_lang(options['data']))
            elif field == 'use_fbank':
                row.append(_get_features(options[field]))
            elif field == 'model':
                row.append(_get_model(options[field]))
            elif field in ['lr', 'lr_drop']:
                row.append('{:.04f}'.format(options[field]))
            elif options[field] is None:
                row.append('no')
            else:
                row.append(str(options[field]))
        row.append('{:.02f}\%'.format(metrics['accuracy']))
        row.append('{:.03f}'.format(metrics['xent']))
        print(' & '.join(row) + ' \\\\')


if __name__ == '__main__':
    main()
