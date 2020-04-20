#!/usr/bin/env bash
set -xueo pipefail

URL=https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz
SRC_DIR="source"
FNAME="data.tar.gz"
RESULT="data"

rm -rf "$SRC_DIR";
mkdir "$SRC_DIR"
wget -O "$SRC_DIR/$FNAME" "$URL"

rm -rf "$RESULT"
mkdir "$RESULT"
tar -xf "$SRC_DIR/$FNAME" -C "$RESULT"
