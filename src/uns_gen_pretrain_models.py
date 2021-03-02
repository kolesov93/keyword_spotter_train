#!/usr/bin/env python3
import numpy as np
import os
import json

np.random.seed(1993)
# ./main.py --lr 0.3 --frame-length 34 --frame-shift 9 --num-mel-bins 32 ../datasets/google_speech_commands/data/ traindir

MODELS = ['res8', 'res15', 'res26_narrow']
BATCH_SIZES = [16, 32, 64, 128]
NS = [20, 100, 500, 1000]

args = []
T = 200

for _ in range(T):
    n = np.random.choice(NS)

    lr = 10 ** np.random.uniform(-3., 0.)
    lr_drop = np.random.uniform(1.1, 10.0)
    dev_every_batches = 2 ** np.random.randint(7, 12)
    batch_size = 2 ** np.random.randint(4, 7)
    use_fbank = True
    model = np.random.choice(MODELS)

    new_args = {
        'n': n,
        'lr': lr,
        'batch-size': batch_size,
        'lr-drop': lr_drop,
        'model': model,
        'dev-every-batches': dev_every_batches,
        'max-batches': '50000',
        'use-fbank': None,
        'self-pretrain': None
    }
    args.append(new_args)

def _make_traindir(args):
    tokens = []
    tokens.append('pretrain' + str(args['n']))
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
    data = 'pretrain' + str(cargs.pop('n'))
    for k, v in cargs.items():
        cmd.append('--' + k)
        if v:
            cmd.append(str(v))

    cmd.append(data)
    cmd.append('uns_pretrain_traindirs/{:03d}_{}'.format(i, traindir))
    cmds.append(' '.join(cmd))

for cmd in cmds:
    print(cmd)
