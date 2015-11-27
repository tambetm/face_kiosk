#!/usr/bin/env python

import caffe
import numpy as np
import cv2
import skimage
import sklearn.preprocessing
import sys

class FaceFeaturesExtractor(object):

    def __init__(self, model_snapshot, model_prototxt, mean_file, image_size, grayscale, oversample, layer, GPU=False):
        if GPU:
            caffe.set_mode_gpu()
        else:
            caffe.set_mode_cpu()

        # create model from prototext file and learned weight values from caffemodel
        self.net = caffe.Net(model_prototxt, model_snapshot, caffe.TEST)

        # load mean blob and discard numsamples and numchannels, because they are 1 anyway.
        if mean_file.endswith(".binaryproto"):
          self.mean = load_blob(mean_file)[0]
          # move channel to last axis
          self.mean = np.transpose(self.mean, (1,2,0))
        elif mean_file.endswith(".npy"):
          self.mean = np.load(mean_file)
        else:
          assert False, "Unknown mean file type"

        # remember image and crop size
        self.image_size = image_size
        self.grayscale = grayscale
        self.oversample = oversample
        self.crop_dims = self.net.blobs["data"].data.shape[2:]
        self.feature_size = self.net.blobs[layer].data.shape[1]
        self.layer = layer

    def get_feature_size(self):
        return self.feature_size
    
    def get_image_features(self, image):
        if self.grayscale:
          image = convert_grayscale(image)
        # resize image to expected size
        image = resize_image(image, self.image_size, self.image_size)
        # scale pixel values from [0,1] to [0,255]
        #image *= 255
        # subtract mean image.
        image = image - self.mean

        if self.oversample:
          # add additional singleton dimension for batch
          image = image[np.newaxis, ...]
          # produce 10 crops per image
          image = caffe.io.oversample(image, self.crop_dims)
        else:
          # crop center
          image = crop_center(image, self.crop_dims[1], self.crop_dims[0])
          # add additional singleton dimension for batch
          image = image[np.newaxis, ...]
        # move channel after batch
        image = image.transpose((0,3,1,2))
        # do feed-forward pass with image as input data, returns data for listed layer
        features = self.net.forward_all([self.layer], data=image)
        # take features from given layer
        features = features[self.layer]
        # reshape is needed only for pooling layer 
        features = np.reshape(features, (features.shape[0], -1))
        # L2-normalize features
        features = sklearn.preprocessing.normalize(features)
        if self.oversample:
          return np.mean(features, axis=0)
        else:
          # take first output of chosen layer, because we supplied only one image
          return features[0]

    def get_face_features_from_file(self, image_filename, face_rect):
        image = load_image(image_filename)
        face = crop_image(image, *face_rect)
        return self.get_image_features(face)

def load_blob(filename):
    # adapted from: https://github.com/BVLC/caffe/issues/1459
    blob = caffe.proto.caffe_pb2.BlobProto()
    data = open(filename, 'rb').read()
    blob.ParseFromString(data)
    return np.array(caffe.io.blobproto_to_array(blob)) # NB! result is NxCxHxW!

def load_image(filename, grayscale = True):
    # adapted from caffe.io.load_image, added converting to grayscale
    image = cv2.imread(filename, CV_LOAD_IMAGE_GRAYSCALE if grayscale else CV_LOAD_IMAGE_COLOR)
    return image # NB! image is HxW, not WxH!

def resize_image(image, w, h):
    if image.shape[:-1] == (h, w):
      return image

    # use OpenCV resize, because it handles BGR images
    image = cv2.resize(image, (h, w))
    # add singleton dimension if grayscale
    if len(image.shape) == 2:
      return image[..., np.newaxis]
    else:
      return image

def convert_grayscale(image):
    if image.shape[-1] == 1:
      return image
      
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # add dummy channel axis
    image = image[..., np.newaxis]
    return image

def crop_image(image, x, y, w, h):
    if image.shape[:-1] == (h, w) and (x, y) == (0, 0):
      return image

    return image[y:(y + h), x:(x + w), :]  # assume image is grayscale HxW

def crop_center(image, w, h):
    if image.shape[:-1] == (h, w):
      return image

    dw = int((image.shape[1] - w) / 2)
    dh = int((image.shape[0] - h) / 2)
    return image[dh:-dh, dw:-dw, :]  # assume image is grayscale HxW
