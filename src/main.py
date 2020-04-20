#!/usr/bin/env python3.8
"""Train keyword spotter."""
import argparse
import logging

import data.google_speech_commands as gsc

LOGGER = logging.getLogger('spotter_train')

DEV_PERCENTAGE = 10.
TEST_PERCENTAGE = 10.


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('data', help='path/to/google_speech_command/dataset')
    return parser.parse_args()


def main(args):
    """Run train."""
    LOGGER.setLevel(logging.INFO)

    sets = gsc.split_index(gsc.get_index(args.data), DEV_PERCENTAGE, TEST_PERCENTAGE)
    for s in sets:
        samples = sum(len(fnames) for fnames in s.values())
        print(samples)

if __name__ == '__main__':
    main(_parse_args())
