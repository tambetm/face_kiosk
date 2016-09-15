#!/usr/bin/env bash

python src/kiosk.py --enlarge_factor=1.4 --image_size=64 --oversample 1 --images_path="images/fotis" --mean_file="models/fotis/lfw+wlf+fotis_train_mean.binaryproto" --data_file="data/fotis/fotis_oversample.csv" --index_file="data/fotis/fotis_oversample.pkl" --model_file="models/fotis/fotis_features.prototxt" --pretrained_file="models/fotis/lfw+wlf+fotis_iter_110000.caffemodel" --layer=ip1
