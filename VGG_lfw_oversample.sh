#!/usr/bin/env bash

python kiosk.py --image_size 256 --average_window=10 --images_path=/storage/LFW/lfw_resized --mean_file=data/vgg_mean.npy --data_file=vgg_lfw/vgg_lfw.csv --index_file=vgg_lfw/vgg_lfw_oversample.pkl --model_file=vgg_face_caffe/VGG_FACE_deploy.prototxt --pretrained_file=vgg_face_caffe/VGG_FACE.caffemodel --layer=fc7 --enlarge_factor=2.2 --backend gpu $*
