#!/bin/bash

TUTORIAL_NUMBER=$1

git clone https://github.com/pythonlessons/mltu.git MLTU
TUTORIAL_DIR=$(find ./MLTU/Tutorials -type d -name "$TUTORIAL_NUMBER*")

find $TUTORIAL_DIR -type f -not -name "README.md" -exec mv {} . \;

rm -rf MLTU