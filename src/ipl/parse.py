#!/usr/bin/env python3
import sys
import numpy as np
import logging

LOGGER = logging.getLogger('parser')
LOGGER_FORMAT = '%(asctime)s - %(pathname)s:%(lineno)d - %(levelname)s - %(message)s'
logging.basicConfig(format=LOGGER_FORMAT, level=logging.DEBUG)

columns = []
ucolumn, scolumn, wcolumn = None, None, None
words = []
word2column = {}
word2num = {}

BEFORE = 3
AFTER = 3
NBEST = 1000
MIN_DIST = 100

LOGGER.info('Starting reading')

P = []
ruttids = []
rshifts = []
for i, line in enumerate(sys.stdin):
    line = line.strip()
    if i == 0:
        columns = line.split()
        ucolumn = columns.index('uttid')
        scolumn = columns.index('shift')
        wcolumn = columns.index('winner')
        for j, word in enumerate(columns):
            if j not in [ucolumn, scolumn, wcolumn]:
                words.append(word)
                word2column[word] = j
        for j, word in enumerate(words):
            word2num[word] = j
        continue
    elif i == 1:
        continue

    values = line.split()
    P.append([float(values[word2column[word]]) for word in words])
    ruttids.append(values[ucolumn])
    rshifts.append(int(values[scolumn]))

    if (i + 1) % 10000 == 0:
        LOGGER.info('Read %d lines', i + 1)

LOGGER.info('Finished reading')

LOGGER.info('Staring computing values')
P = np.array(P)
nrows = len(P)
nwords = len(words)
word2values = [[] for _ in range(nwords)]
for i in range(BEFORE, nrows - AFTER):
    for j in range(nwords):
        value = np.min(P[i-BEFORE:i+AFTER, j])
        word2values[j].append((value, i))

    if (i + 1) % 10000 == 0:
        LOGGER.info('Computed values for %d/%d lines', i + 1, nrows)

LOGGER.info('Finished computing values')
LOGGER.info('Started sorting values')

for j in range(nwords):
    word2values[j].sort(reverse=True)

LOGGER.info('Finished sorting values')

for j, word in enumerate(words):
    print(word)
    taken_poses = []
    for value, pos in word2values[j]:
        cool = True
        for taken_pos in taken_poses:
            if abs(taken_pos - pos) < MIN_DIST:
                cool = False
                break

        if not cool:
            continue

        print('\t{}\t{}\t{:.2f}'.format(ruttids[pos], rshifts[pos], value))
        taken_poses.append(pos)
        if len(taken_poses) == NBEST:
            break
