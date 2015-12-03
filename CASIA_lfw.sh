#!/usr/bin/env bash

python src/kiosk.py --enlarge_factor=2.2 --image_size=100 --images_path=images/lfw --mean_file=models/CASIA/CASIA_train_mean.binaryproto --data_file=data/CASIA_lfw/lfw_100_all.csv --index_file=data/CASIA_lfw/lfw_100_all.pkl --model_file=models/CASIA/CASIA_features.prototxt --pretrained_file=models/CASIA/CASIA_iter_450000.caffemodel --layer=pool5 --backend gpu $*
