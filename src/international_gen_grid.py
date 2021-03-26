#!/usr/bin/env python3
import numpy as np
import os

np.random.seed(1993)

LANGUAGES = {
    'lten_data_only_lt_words': {
        'words': 'ne,ačiū,stop,įjunk,išjunk,į_viršų,į_apačią,į_dešinę,į_kairę,startas,pauzė,labas,iki',
        'data': '/home/kolesov93/study/keyword_spotter_train/src/international/sets/lten_data'
    },
    'lten_data': {
        'words': 'yes,no,up,down,left,right,on,off,stop,go,ne,ačiū,stop,įjunk,išjunk,į_viršų,į_apačią,į_dešinę,į_kairę,startas,pauzė,labas,iki',
        'data': '/home/kolesov93/study/keyword_spotter_train/src/international/sets/lten_data'
    },
    'lt': {
        'words': 'ne,ačiū,stop,įjunk,išjunk,į_viršų,į_apačią,į_dešinę,į_kairę,startas,pauzė,labas,iki',
        'data': '/home/kolesov93/study/datasets/lt_data'
    }
}

MODELS = ['ff', 'res8', 'res8_narrow', 'res15', 'res15_narrow', 'res26', 'res26_narrow']

args = []
T = 200

for _ in range(T):
    model = np.random.choice(MODELS)
    lr = 10 ** np.random.uniform(-3., 0.)
    lr_drop = np.random.uniform(1.1, 10.0)
    dev_every_batches = 2 ** np.random.randint(7, 12)
    batch_size = 2 ** np.random.randint(4, 7)
    use_fbank = True
    for lang in LANGUAGES:
        args.append({
            'lr': lr,
            'batch-size': batch_size,
            'wanted-words': LANGUAGES[lang]['words'],
            'language': lang,
            'lr-drop': lr_drop,
            'model': model,
            'dev-every-batches': dev_every_batches,
            'use-fbank': None
        })

def _make_traindir(args):
    tokens = []
    tokens.append(args['language'])
    tokens.append(args['model'])
    return '_'.join(tokens)

cmds = []
for i, cargs in enumerate(args):
    cmd = ['PYTHONPATH=/home/kolesov93/study/fairseq', 'python3.6', './main.py']
    traindir = _make_traindir(cargs)
    language = cargs.pop('language')
    for k, v in cargs.items():
        cmd.append('--' + k)
        if v:
            cmd.append(str(v))

    cmd.append(LANGUAGES[language]['data'])
    cmd.append('international/grid1/{:03d}_{}'.format(i, traindir))
    cmds.append(' '.join(cmd))

for cmd in cmds:
    print(cmd)
