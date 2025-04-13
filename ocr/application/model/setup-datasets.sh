#!/bin/bash

# Set the target directory for unpacking
DATASETS_DIR="datasets/iam/"

# Make the directories if necessary
mkdir -p $DATASETS_DIR

# Unpack
tar -xf lines.tgz -C $DATASETS_DIR

# Performing mv after unzip seems easier than unpacking without creating dir
unzip -qq archive.zip iam_dataset/* -d $DATASETS_DIR
mv $DATASETS_DIR/iam_dataset/* $DATASETS_DIR
rm -r $DATASETS_DIR/iam_dataset
