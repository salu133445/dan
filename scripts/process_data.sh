#!/bin/bash
# This script processes and saves the training data to shared memory.
# Usage: process_data.sh
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" > /dev/null && pwd )"
python3 "$DIR/../src/process_data.py" --make_imbalanced \
  --make_super_imbalanced "$DIR/../data/mnist/"
