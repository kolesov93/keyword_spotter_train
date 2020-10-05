#!/usr/bin/env python3
import numpy as np
import os

np.random.seed(1993)
# ./main.py --lr 0.3 --frame-length 34 --frame-shift 9 --num-mel-bins 32 ../datasets/google_speech_commands/data/ traindir
WAV2VEC_MODEL_PATH = '/home/kolesov93/study/wav2vec_models/first.pt'

LANGUAGES = {
    'ru': {
        'words': 'один,два,три,четыре,пять,да,нет,спасибо,стоп,включи',
        'data': '/home/kolesov93/study/datasets/rus_data'
    },
    'en': {
        'words': 'yes,no,up,down,left,right,on,off,stop,go',
        'data': '/home/kolesov93/study/datasets/data'
    }
}

MODELS = ['ff', 'res8', 'res8_narrow', 'res15', 'res15_narrow', 'res26', 'res26_narrow']
LIMITS = [None, 3, 5, 7, 10, 20]
LRS = [0.1, 0.01]
BATCH_SIZES = [32, 64]

args = []
T = 100

for limit in LIMITS:
    for model in MODELS:
        for lang in LANGUAGES:
            for lr in LRS:
                for batch_size in BATCH_SIZES:
                    for use_fbank in [False, True]:
                        new_args = {
                            'lr': lr,
                            'batch-size': batch_size,
                            'model-path': WAV2VEC_MODEL_PATH,
                            'wanted-words': LANGUAGES[lang]['words'],
                            'language': lang,
                            'model': model
                        }
                        if limit is not None:
                            new_args['limit'] = limit
                        if use_fbank:
                            new_args['use-fbank'] = None
                        args.append(new_args)

def _make_traindir(args):
    tokens = []
    tokens.append(args['language'])
    if 'use-fbank' in args:
        tokens.append('fbank')
    else:
        tokens.append('wav2vec')
    tokens.append(args['model'])
    if 'limit' in args:
        tokens.append('limit{}'.format(args['limit']))
    tokens.append('batch{}'.format(args['batch-size']))
    tokens.append('lr{}'.format(args['lr']))
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
    cmd.append('traindirs/{:03d}_{}'.format(i, traindir))
    cmds.append(' '.join(cmd))

for cmd in cmds:
    print(cmd)
