#!/usr/bin/env python3
import sys
import os
import time
import random
import subprocess as sp
import multiprocessing
import shutil
import faulthandler
import signal


class MySigtermError(Exception):
    pass


def f(cmd):
    ocmd = cmd
    gpu = f.gpu_queue.get()
    good = False
    traindir = cmd.split()[-1]
    T = 6
    cmd = 'CUDA_VISIBLE_DEVICES={} '.format(gpu) + cmd
    for _ in range(T):
        try:
            proc = sp.Popen(cmd, shell=True)
            returncode = proc.wait()
            if returncode == 0:
                good = True
                break
            raise sp.CalledProcessError
        except sp.CalledProcessError:
            shutil.rmtree(traindir)
            time.sleep(10)
        except MySigtermError:
            print('Caught SIGTERM exception', file=sys.stderr)
            if proc.poll() is None:
                print('Proc is still alive. Terminating', file=sys.stderr)
                proc.terminate()
                print('Terminated', file=sys.stderr)
                assert proc.pool() is not None
            shutil.rmtree(traindir)
            break
    f.gpu_queue.put(gpu)
    return ocmd, good


def sigterm_handler(signum, frame):
    print('Caught SIGTERM', file=sys.stderr)
    raise MySigtermError()


def f_init(gpu_queue):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, sigterm_handler)
    f.gpu_queue = gpu_queue


def main():
    faulthandler.enable()
    GPUS = [0]
    cmds = [line.strip() for line in sys.stdin]

    gpu_queue = multiprocessing.Queue()
    for gpu in GPUS:
        gpu_queue.put(gpu)

    nprocesses = len(GPUS)
    with multiprocessing.Pool(nprocesses, f_init, [gpu_queue]) as pool, open('result.txt', 'w') as fout:
        try:
            for cmd, good in pool.imap_unordered(f, cmds):
                print('{}\t{}'.format(good, cmd), file=fout)
                fout.flush()
        except KeyboardInterrupt:
            pool.terminate()


if __name__ == '__main__':
    main()
