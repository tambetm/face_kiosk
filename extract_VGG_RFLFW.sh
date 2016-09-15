#!/usr/bin/env bash

python src/extract.py images/RFLFW data/VGG_RFLFW/VGG_RFLFW.npz data/VGG_RFLFW/VGG_RFLFW.csv --image_size=256 --grayscale=0 --mean_file=models/vgg_mean.npy --model_file=models/vgg_face_caffe/VGG_FACE_deploy.prototxt --pretrained_file=models/vgg_face_caffe/VGG_FACE.caffemodel --layer=fc7 --backend gpu $*

python src/build_index.py data/VGG_RFLFW/VGG_RFLFW.npz data/VGG_RFLFW/VGG_RFLFW.pkl
