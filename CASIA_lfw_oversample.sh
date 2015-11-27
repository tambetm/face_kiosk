#!/usr/bin/env bash

python kiosk.py --image_size 100 --average_window 10 --images_path=/storage/LFW/lfw_resized --mean_file=CASIA_lfw/CASIA_train_mean.binaryproto --data_file=CASIA_lfw/CASIA_lfw_oversample.csv --index_file=CASIA_lfw/CASIA_lfw_oversample.pkl --model_file=CASIA_lfw/CASIA_features.prototxt --pretrained_file=CASIA_lfw/CASIA_iter_450000.caffemodel --layer=pool5 --enlarge_factor=2.2 --backend gpu $*
