#!/usr/bin/env bash

python src/extract.py images/lfw data/CASIA_lfw/CASIA_lfw_oversample.npz data/CASIA_lfw/CASIA_lfw_oversample.csv --image_size 100 --grayscale 1 --mean_file=models/CASIA/CASIA_train_mean.binaryproto --model_file=models/CASIA/CASIA_features.prototxt --pretrained_file=models/CASIA/CASIA_iter_450000.caffemodel --layer=pool5 --oversample 1 --backend gpu $*

python src/build_index.py data/CASIA_lfw/CASIA_lfw_oversample.npz data/CASIA_lfw/CASIA_lfw_oversample.pkl

