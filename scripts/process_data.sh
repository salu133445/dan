#!/bin/bash
# This script store the training data to shared memory.
# Usage: process_data.sh
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" > /dev/null && pwd )"
python3 "$DIR/../src/process_data.py" "$DIR/../data/mnist/"
