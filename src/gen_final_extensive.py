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
    },
    'lt': {
        'words': 'nulis,vienas,du,trys,keturi,penki,taip,ne,ačiū,stop,įjunk,išjunk,į_viršų_,į_apačią,į_dešinę,į_kairę,startas,pauzė,labas,iki',
        'data': '/home/kolesov93/study/datasets/rus_data'
    }
}

MODELS = ['ff', 'res8', 'res8_narrow', 'res15', 'res15_narrow', 'res26', 'res26_narrow']
LIMITS = [None, 3, 5, 7, 10, 20]
LRS = [0.1, 0.01]
BATCH_SIZES = [16, 32, 64, 128]

args = []
T = 2000

for _ in range(T):
    limit = np.random.choice(LIMITS)
    model = np.random.choice(MODELS)
    lr = 10 ** np.random.uniform(-3., 0.)
    lr_drop = np.random.uniform(1.1, 10.0)
    dev_every_batches = 2 ** np.random.randint(3, 12)
    batch_size = 2 ** np.random.randint(4, 7)
    lang = np.random.choice(list(LANGUAGES.keys()))
    use_fbank = np.random.choice([False, True])

    if np.random.randint(0, 100) < 25:
        limit = None
        lang = 'ru'
        model = np.random.choice(['res8', 'res8_narrow', 'res15', 'res15_narrow'])
        if model == 'res8':
            use_fbank = True
        else:
            use_fbank = False
    else:
        model = np.random.choice(['res8', 'res8_narrow', 'res15', 'res15_narrow'])
        lang = np.random.choice(['ru', 'en'])
        limit = np.random.choice([15, 25, 30, 50, 75, 100])

    new_args = {
        'lr': lr,
        'batch-size': batch_size,
        'model-path': WAV2VEC_MODEL_PATH,
        'wanted-words': LANGUAGES[lang]['words'],
        'language': lang,
        'lr-drop': lr_drop,
        'model': model,
        'dev-every-batches': dev_every_batches
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
    tokens.append('lr_drop{}'.format(args['lr-drop']))
    tokens.append('dev_every_batches{}'.format(args['dev-every-batches']))
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
