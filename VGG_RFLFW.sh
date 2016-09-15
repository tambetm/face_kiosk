#!/usr/bin/env bash

python src/kiosk.py --enlarge_factor=1.4 --image_size=256 --grayscale=0 --images_path=images/RFLFW --mean_file=models/vgg_mean.npy --data_file=data/VGG_RFLFW/VGG_RFLFW.csv --index_file=data/VGG_RFLFW/VGG_RFLFW.pkl --model_file=models/vgg_face_caffe/VGG_FACE_deploy.prototxt --pretrained_file=models/vgg_face_caffe/VGG_FACE.caffemodel --layer=fc7 --backend gpu $*
