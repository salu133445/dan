#!/bin/bash
# This script downloads the MNIST handwritten digit dataset to the default data
# diretory.
# Usage: download_data.sh
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" > /dev/null && pwd )"
DST="${DIR}/../data/mnist/"
mkdir -p "$DST"
wget -P "$DST" "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz"
wget -P "$DST" "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz"
wget -P "$DST" "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz"
wget -P "$DST" "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz"
gunzip "$DST"/*.gz
