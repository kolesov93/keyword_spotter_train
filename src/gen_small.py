#!/usr/bin/env python3
import numpy as np
import os

np.random.seed(1993)
# ./main.py --lr 0.3 --frame-length 34 --frame-shift 9 --num-mel-bins 32 ../datasets/google_speech_commands/data/ traindir
model_path = '/home/kolesov93/study/wav2vec_models/first.pt'

args = []
T = 200
for _ in range(T):
    new_args = {
        'lr': 10 ** np.random.uniform(-2., 0.),
        'batch-size': 2**np.random.randint(6, 8),
        'limit': np.random.choice([10]),
        'model-path': model_path
    }
    if np.random.randint(0, 100) < 50:
        new_args['use-fbank'] = None
    if np.random.randint(0, 100) < 50:
        new_args['use-resnet'] = None
    args.append(new_args)

def _make_traindir(args):
    tokens = []
    if 'use-fbank' in args:
        tokens.append('fbank')
    else:
        tokens.append('wav2vec')
    if 'use-resnet' in args:
        tokens.append('resnet')
    else:
        tokens.append('ff')
    tokens.append('limit{}'.format(args['limit']))
    tokens.append('batch{}'.format(args['batch-size']))
    return '_'.join(tokens)

cmds = []
for i, cargs in enumerate(args):
    cmd = ['PYTHONPATH=/home/kolesov93/study/fairseq', 'python3.6', './main.py']
    for k, v in cargs.items():
        cmd.append('--' + k)
        if v:
            cmd.append(str(v))

    cmd.append('/home/kolesov93/study/datasets/data')
    cmd.append('small_traindirs/{:03d}_{}'.format(i, _make_traindir(cargs)))
    cmds.append(' '.join(cmd))

for cmd in cmds:
    print(cmd)
