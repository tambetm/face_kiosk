#!/usr/bin/env bash

python src/kiosk.py --enlarge_factor=2.2 --image_size=100 --oversample=1 --images_path=images/CASIA --mean_file=models/CASIA/CASIA_train_mean.binaryproto --data_file=data/CASIA/CASIA.csv --index_file=data/CASIA/CASIA_oversample.pkl --model_file=models/CASIA_lfw/CASIA_features.prototxt --pretrained_file=models/CASIA_lfw/CASIA_iter_450000.caffemodel --layer=pool5 --backend gpu $*
