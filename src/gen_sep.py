#!/usr/bin/env python3
import numpy as np
import os

np.random.seed(1993)
# ./main.py --lr 0.3 --frame-length 34 --frame-shift 9 --num-mel-bins 32 ../datasets/google_speech_commands/data/ traindir

args = []
T = 200
for _ in range(T):
    new_args = {
        'lr': 10 ** np.random.uniform(-2., 0.),
        'batch-size': 2**np.random.randint(6, 8),
        'model-path': '/home/kolesov93/study/wav2vec_models/{}.pt'.format(np.random.randint(0, 11))
    }
    coin = np.random.randint(0, 100)
    if coin < 50:
        new_args['use-fbank'] = None
        new_args['use-wav2vec'] = None
    elif coin < 75:
        new_args['use-fbank'] = None
    else:
        new_args['use-wav2vec'] = None
    args.append(new_args)

cmds = []
for i, cargs in enumerate(args):
    cmd = ['PYTHONPATH=/home/kolesov93/study/fairseq', 'python3.6', './main.py']
    for k, v in cargs.items():
        cmd.append('--' + k)
        if v:
            cmd.append(str(v))
    cmd.append('/home/kolesov93/study/datasets/data')
    cmd.append('wav2vec_sep_traindirs/{:03d}'.format(i))
    cmds.append(' '.join(cmd))

for cmd in cmds:
    print(cmd)
