#!/usr/bin/env bash

python kiosk.py --enlarge_factor=2.2 --image_size=100 --images_path=/storage/LFW/lfw_resized --mean_file=CASIA_lfw/CASIA_train_mean.binaryproto --data_file=CASIA_lfw/lfw_100_all.csv --index_file=CASIA_lfw/lfw_100_all.pkl --model_file=CASIA_lfw/CASIA_features.prototxt --pretrained_file=CASIA_lfw/CASIA_iter_450000.caffemodel --layer=pool5 --backend gpu $*
