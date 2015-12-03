#!/usr/bin/env bash

python src/extract.py images/fotis data/fotis/fotis.npz data/fotis/fotis.csv --image_size 64 --grayscale 1 --mean_file=models/fotis/lfw+wlf+fotis_train_mean.binaryproto --model_file=models/fotis/fotis_features.prototxt --pretrained_file=models/fotis/lfw+wlf+fotis_iter_110000.caffemodel --layer=ip1 --backend gpu $*

python src/build_index.py data/fotis/fotis.npz data/fotis/fotis.pkl

