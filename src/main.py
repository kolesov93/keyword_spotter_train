#!/usr/bin/env python3.8
"""Train keyword spotter."""
import argparse
import logging

import sounddevice as sd

import data.google_speech_commands as gsc

LOGGER = logging.getLogger('spotter_train')

DEV_PERCENTAGE = 10.
TEST_PERCENTAGE = 10.

WANTED_WORDS = 'yes,no,up,down,left,right,on,off,stop,go'.split(',')


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('data', help='path/to/google_speech_command/dataset')
    return parser.parse_args()


def main(args):
    """Run train."""
    LOGGER.setLevel(logging.INFO)

    indexes = gsc.split_index(gsc.get_index(args.data), DEV_PERCENTAGE, TEST_PERCENTAGE)
    testset = gsc.GoogleSpeechCommandsDataset(indexes[2], WANTED_WORDS)
    for sample in testset:
        print(sample['fname'], gsc._get_label(sample['label'], WANTED_WORDS))
        sd.play(sample['data'], gsc.SAMPLE_RATE, blocking=True)

if __name__ == '__main__':
    main(_parse_args())
