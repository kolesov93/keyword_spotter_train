#!/usr/bin/env python3
import sys
import os
import time
import random
import subprocess as sp
import multiprocessing
import shutil
import faulthandler


def f(cmd):
    ocmd = cmd
    gpu = f.gpu_queue.get()
    good = False
    traindir = cmd.split()[-1]
    T = 6
    for _ in range(T):
        try:
            cmd = 'CUDA_VISIBLE_DEVICES={} '.format(gpu) + cmd
            sp.check_call(cmd, shell=True)
            good = True
            break
        except:
            shutil.rmtree(traindir)
            time.sleep(10)
    f.gpu_queue.put(gpu)
    return ocmd, good


def f_init(gpu_queue):
    f.gpu_queue = gpu_queue


def main():
    faulthandler.enable()
    GPUS = list(range(8))
    cmds = [line.strip() for line in sys.stdin]

    gpu_queue = multiprocessing.Queue()
    for gpu in GPUS:
        gpu_queue.put(gpu)

    nprocesses = len(GPUS)
    with multiprocessing.Pool(nprocesses, f_init, [gpu_queue]) as pool, open('result.txt', 'w') as fout:
        for cmd, good in pool.imap_unordered(f, cmds):
            print('{}\t{}'.format(good, cmd), file=fout)
            fout.flush()


if __name__ == '__main__':
    main()

