#!/usr/bin/env python3
import numpy as np
import os
import json

np.random.seed(1993)

MODELS = ['res8', 'res15', 'res26_narrow']
BATCH_SIZES = [16, 32, 64, 128]

args = []
T = 200

for _ in range(T):
    lr = 10 ** np.random.uniform(-3., 0.)
    lr_drop = np.random.uniform(1.1, 10.0)
    specaug_level = np.random.choice([1, 2, 3])
    dev_every_batches = 2 ** np.random.randint(7, 12)
    batch_size = 2 ** np.random.randint(4, 7)
    use_fbank = True
    model = np.random.choice(MODELS)

    new_args = {
        'lr': lr,
        'batch-size': batch_size,
        'specaug-level': specaug_level,
        'lr-drop': lr_drop,
        'model': model,
        'dev-every-batches': dev_every_batches,
        'max-batches': '100000',
        'use-fbank': None,
        'self-pretrain': None
    }
    args.append(new_args)

def _make_traindir(args):
    tokens = []
    tokens.append('specaug' + str(args['specaug-level']))
    tokens.append(args['model'])
    tokens.append('batch{}'.format(args['batch-size']))
    tokens.append('lr{}'.format(args['lr']))
    tokens.append('lr_drop{}'.format(args['lr-drop']))
    tokens.append('dev_every_batches{}'.format(args['dev-every-batches']))
    return '_'.join(tokens)

cmds = []
for i, cargs in enumerate(args):
    cmd = ['PYTHONPATH=/home/kolesov93/study/fairseq', 'python3.6', './main.py']
    traindir = _make_traindir(cargs)
    for k, v in cargs.items():
        cmd.append('--' + k)
        if v:
            cmd.append(str(v))

    data = '/home/kolesov93/study/datasets/lt_pretrain'
    cmd.append(data)
    cmd.append('uns_lt_pretrain_traindirs/{:03d}_{}'.format(i, traindir))
    cmds.append(' '.join(cmd))

for cmd in cmds:
    print(cmd)
