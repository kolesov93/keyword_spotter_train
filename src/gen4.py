#!/usr/bin/env python3
import numpy as np
import os

np.random.seed(1993)
# ./main.py --lr 0.3 --frame-length 34 --frame-shift 9 --num-mel-bins 32 ../datasets/google_speech_commands/data/ traindir

args = []
T = 200
for _ in range(T):
    args.append({
        'lr': 10 ** np.random.uniform(-3., 0.),
        'batch-size': 2**np.random.randint(5, 10),
        'model-path': '/home/kolesov93/study/wav2vec_models/{}.pt'.format(np.random.randint(0, 11))
    })

cmds = []
for i, cargs in enumerate(args):
    cmd = ['PYTHONPATH=/home/kolesov93/study/fairseq', 'python3.6', './main.py']
    for k, v in cargs.items():
        cmd.append('--' + k)
        cmd.append(str(v))
    cmd.append('/home/kolesov93/study/datasets/data')
    cmd.append('wav2vec_concat_traindirs/{:03d}'.format(i))
    cmds.append(' '.join(cmd))

for cmd in cmds:
    print(cmd)
