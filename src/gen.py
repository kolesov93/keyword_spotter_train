#!/usr/bin/env python3
import numpy as np

np.random.seed(1993)
# ./main.py --lr 0.3 --frame-length 34 --frame-shift 9 --num-mel-bins 32 ../datasets/google_speech_commands/data/ traindir

args = []
for lr in [0.05, 0.1, 0.2]:
    for frame_length in [10, 25, 30]:
        for frame_shift in [10]:
            for bins in [10, 20, 40, 60, 80, 100, 120]:
                args.append(
                    {'lr': lr, 'frame-length': frame_length, 'frame-shift': frame_shift, 'num-mel-bins': bins}
                )

    for frame_length in np.arange(5, 50, 2):
        for frame_shift in [10]:
            for bins in [40, 80]:
                args.append(
                    {'lr': lr, 'frame-length': frame_length, 'frame-shift': frame_shift, 'num-mel-bins': bins}
                )

    for frame_length in [25]:
        for frame_shift in np.arange(5, 50, 2):
            for bins in [40, 80]:
                args.append(
                    {'lr': lr, 'frame-length': frame_length, 'frame-shift': frame_shift, 'num-mel-bins': bins}
                )

T = 200
for _ in range(T):
    lr = 10. ** np.random.uniform(-2., 0.)
    frame_length = np.random.randint(5, 60)
    frame_shift = np.random.randint(5, frame_length + 1)
    bins = np.random.randint(20, 120)
    args.append(
        {'lr': lr, 'frame-length': frame_length, 'frame-shift': frame_shift, 'num-mel-bins': bins}
    )

cmds = []
for i, cargs in enumerate(args):
    cmd = ['./main.py']
    for k, v in cargs.items():
        cmd.append('--' + k)
        cmd.append(str(v))
    cmd.append('../datasets/google_speech_commands/data/')
    cmd.append('traindirs/{:03d}'.format(i))
    cmds.append(' '.join(cmd))

for cmd in cmds:
    print(cmd)
