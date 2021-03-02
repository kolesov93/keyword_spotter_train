#!/usr/bin/env python3
import argparse
import json
import sys
import os

class FieldGetter:

    def __init__(self, field):
        self._field = field

    def get_value(self, options):
        return options[self._field]

    def get_formatted_value(self, options):
        return str(self.get_value(options))


class LangGetter:

    def get_value(self, options):
        data = options['data']
        if 'rus_data' in data:
            return 'ru'
        elif 'lt_data' in data:
            return 'lt'
        return 'en'

    def get_formatted_value(self, options):
        return self.get_value(options)


class UptrainGetter:

    def get_value(self, options):
        if options.get('initialize_body') is not None:
            return 'yes'
        return 'no'

    def get_formatted_value(self, options):
        return self.get_value(options)


class FeaturesGetter:

    def get_value(self, options):
        if options['use_fbank']:
            return 'fbank'
        return 'wav2vec'

    def get_formatted_value(self, options):
        return self.get_value(options)


class ModelGetter:

    def get_value(self, options):
        return options['model'].replace('_', '-')

    def get_formatted_value(self, options):
        return '\\texttt{' + self.get_value(options) + '}'


class FloatGetter:

    def __init__(self, field, frmt):
        self._field = field
        self._format = frmt

    def get_value(self, options):
        return options[self._field]

    def get_formatted_value(self, options):
        return self._format.format(self.get_value(options))

class PossibleNoneFieldGetter:

    def __init__(self, field, default_value=None, default_formatted_value=None):
        self._field = field
        self._default_value = default_value
        self._default_formatted_value = default_formatted_value

    def get_value(self, options):
        return options[self._field] or self._default_value

    def get_formatted_value(self, options):
        return str(options[self._field] or self._default_formatted_value)


GETTERS = {
    'limit': PossibleNoneFieldGetter('limit', -1, 'no'),
    'uptrain': UptrainGetter(),
    'features': FeaturesGetter(),
    'batch_size': FieldGetter('batch_size'),
    'dev_every_batches': FieldGetter('dev_every_batches'),
    'model': ModelGetter(),
    'lr': FloatGetter('lr', '{:.04f}'),
    'lr_drop': FloatGetter('lr_drop', '{:.04f}'),
    'lang': LangGetter(),
    'accuracy': FloatGetter('accuracy', '{:.02f}%'),
    'xent': FloatGetter('xent', '{:.03f}')
}
FIELDS = ['limit', 'uptrain', 'features', 'batch_size', 'dev_every_batches', 'model', 'lr', 'lr_drop', 'lang']
METRICS = ['accuracy', 'xent']


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
    return tuple(GETTERS[key].get_value(options) for key in keys)


def _get_lang(data):
    if 'rus_data' in data:
        return 'ru'
    if 'lt_data' in data:
        return 'lt'
    return 'en'


def _get_uptrain(data):
    if 'initialize_body' in data:
        return 'yes'
    return 'no'


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

        for field in FIELDS:
            try:
                GETTERS[field].get_value(options)
            except Exception as ex:
                print("Can't get value '{}' from {}".format(field, fname), file=sys.stderr)
                continue

        for metric in METRICS:
            try:
                GETTERS[metric].get_value(metrics)
            except Exception as ex:
                print("Can't get metric '{}' from {}".format(metric, fname), file=sys.stderr)
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
        row = FIELDS + METRICS
        print(' & '.join(row) + ' \\\\')

    for key in sorted(key2best):
        options, metrics = key2best[key]
        row = []
        for field in FIELDS:
            row.append(GETTERS[field].get_formatted_value(options))
        for metric in METRICS:
            row.append(GETTERS[metric].get_formatted_value(metrics))
        print(' & '.join(row) + ' \\\\')


if __name__ == '__main__':
    main()
