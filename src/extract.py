import argparse
import os
import fnmatch
import csv
import sys
import numpy as np
import cv2
from extractor import *

parser = argparse.ArgumentParser()
parser.add_argument("input_path")
parser.add_argument("features_file")
parser.add_argument("text_file")
parser.add_argument("--model_file", default="vgg_face_caffe/VGG_FACE_deploy.prototxt")
parser.add_argument("--pretrained_file", default="vgg_face_caffe/VGG_FACE.caffemodel")
parser.add_argument("--mean_file", default="vgg_face_caffe/vgg_mean.npy")
parser.add_argument("--image_size", type=int, default=256)
parser.add_argument("--oversample", type=int, default=1)
parser.add_argument("--grayscale", type=int, default=0)
parser.add_argument("--layer", default="fc7")
parser.add_argument("--filter", default="*.jpg")
parser.add_argument("--backend", choices=["gpu", "cpu"], default="cpu")
args = parser.parse_args()

print "Scanning files..."
files = []
for dir, subdirs, filenames in os.walk(args.input_path):
  print dir, len(filenames)
  for file in fnmatch.filter(filenames, args.filter):
    files.append(os.path.join(dir, file))
print "Found %d files" % len(files)

extractor = FaceFeaturesExtractor(
    model_snapshot=args.pretrained_file, 
    model_prototxt=args.model_file, 
    mean_file=args.mean_file, 
    image_size=args.image_size,
    grayscale=(args.grayscale==1),
    oversample=(args.oversample==1),
    layer=args.layer, 
    GPU=(args.backend=="gpu"))

feature_size = extractor.get_feature_size()
features = np.empty((len(files), feature_size))

print "Extracting features..."
csv_file = open(args.text_file, "wb")
csv_writer = csv.writer(csv_file)
for i, filename in enumerate(files):
  print i, filename

  image = cv2.imread(os.path.join(args.input_path, filename))
  features[i, :] = extractor.get_image_features(image)

  relname = os.path.relpath(filename, args.input_path)
  dirname = os.path.dirname(filename)
  basename = os.path.basename(dirname)
  name = basename.replace('_', ' ')
  csv_writer.writerow([i + 1, relname, name])

np.savez_compressed(args.features_file, data=features)
csv_file.close()
print "Finished..."
