#!/usr/bin/env python3
import numpy as np
import os
import json

np.random.seed(1993)

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
        'words': 'ne,ačiū,stop,įjunk,išjunk,į_viršų,į_apačią,į_dešinę,į_kairę,startas,pauzė,labas,iki',
        'data': '/home/kolesov93/study/datasets/lt_data'
    }
}

MDLS = """
uns_pretrain_traindirs/005_pretrain2000_specaug3_res15_batch64_lr0.08598910034826232_lr_drop9.640775422935539_dev_every_batches1024/final.mdl
uns_pretrain_traindirs/018_pretrain2000_specaug2_res8_batch32_lr0.6590548709383639_lr_drop3.505846508098238_dev_every_batches2048/final.mdl
uns_pretrain_traindirs/001_pretrain2000_specaug1_res15_batch16_lr0.15038571301597353_lr_drop6.180555034057818_dev_every_batches1024/final.mdl
uns_pretrain_traindirs/009_pretrain100_specaug2_res8_batch64_lr0.0010938016689317363_lr_drop3.6637493358333826_dev_every_batches1024/final.mdl
uns_pretrain_traindirs/016_pretrain100_specaug1_res8_batch64_lr0.021694113785292814_lr_drop8.03090685980859_dev_every_batches256/final.mdl
uns_pretrain_traindirs/010_pretrain100_specaug2_res8_batch64_lr0.019993866427924036_lr_drop7.744913484896234_dev_every_batches256/final.mdl
uns_pretrain_traindirs/002_pretrain100_specaug3_res8_batch32_lr0.0458282164476626_lr_drop2.0591053331402005_dev_every_batches1024/final.mdl
uns_pretrain_traindirs/021_pretrain100_specaug3_res8_batch16_lr0.032422743046281205_lr_drop8.15471858973088_dev_every_batches2048/final.mdl
uns_pretrain_traindirs/020_pretrain1000_specaug3_res8_batch16_lr0.78043050709315_lr_drop4.607502053032845_dev_every_batches2048/final.mdl
""".strip().split()

LIMITS = [None, 3, 5, 7, 10, 20]

args = []
T = 100

for _ in range(T):
    init_model = np.random.choice(MDLS)
    limit = np.random.choice(LIMITS)

    lr = 10 ** np.random.uniform(-3., 0.)
    lr_drop = np.random.uniform(1.1, 10.0)
    dev_every_batches = 2 ** np.random.randint(7, 12)
    batch_size = 2 ** np.random.randint(4, 7)

    lang = 'lt'
    use_fbank = True

    path_tokens = init_model.split('/')
    with open('/'.join(path_tokens[:-1] + ['options.json'])) as fin:
        model = json.load(fin)['model']

    new_args = {
        'lr': lr,
        'batch-size': batch_size,
        'wanted-words': LANGUAGES[lang]['words'],
        'language': lang,
        'lr-drop': lr_drop,
        'model': model,
        'dev-every-batches': dev_every_batches,
        'max-batches': '10000',
        'use-fbank': None,
    }
    if np.random.uniform() < 0.5:
        new_args['initialize-body'] = init_model
    if limit is not None:
        new_args['limit'] = limit
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
    cmd.append('uns_lt_uptrain_after_pretrain_traindirs/{:03d}_{}'.format(i, traindir))
    cmds.append(' '.join(cmd))

for cmd in cmds:
    print(cmd)
