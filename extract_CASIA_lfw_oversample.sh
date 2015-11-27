#!/usr/bin/env bash

python extract.py /storage/LFW/lfw CASIA_lfw/CASIA_lfw_oversample.npz CASIA_lfw/CASIA_lfw_oversample.csv --image_size 100 --grayscale 1 --mean_file=CASIA_lfw/CASIA_train_mean.binaryproto --model_file=CASIA_lfw/CASIA_features.prototxt --pretrained_file=CASIA_lfw/CASIA_iter_450000.caffemodel --layer=pool5 --oversample 1 --backend gpu $*
