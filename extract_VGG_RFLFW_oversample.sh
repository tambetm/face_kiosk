#!/usr/bin/env bash

python src/extract.py images/RFLFW data/VGG_RFLFW/VGG_RFLFW_oversample.npz data/VGG_RFLFW/VGG_RFLFW.csv --oversample 1 --image_size=256 --grayscale=0 --mean_file=models/vgg_mean.npy --model_file=models/vgg_face_caffe/VGG_FACE_deploy.prototxt --pretrained_file=models/vgg_face_caffe/VGG_FACE.caffemodel --layer=fc7 --backend gpu $*

python src/build_index.py data/VGG_RFLFW/VGG_RFLFW_oversample.npz data/VGG_RFLFW/VGG_RFLFW_oversample.pkl
