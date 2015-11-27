#!/usr/bin/env bash

python kiosk.py --enlarge_factor=2.2 --image_size=100 --oversample=1 --images_path=/storage/CASIA-WebFace --mean_file=CASIA_lfw/CASIA_train_mean.binaryproto --data_file=CASIA/CASIA.csv --index_file=CASIA/CASIA_oversample.pkl --model_file=CASIA_lfw/CASIA_features.prototxt --pretrained_file=CASIA_lfw/CASIA_iter_450000.caffemodel --layer=pool5 --backend gpu $*
