# adapted from https://github.com/shantnu/FaceDetect/

import sys
import numpy as np
import cv2

OPENCV_HAAR_CASCADE_FRONTALFACE = 'data/haarcascade_frontalface_default.xml'
OPENCV_MIN_NEIGHBORS = 5
OPENCV_SCALE_FACTOR = 1.1
OPENCV_MIN_SIZE = 32
ENLARGE_FACTOR = 1.4

class FaceDetector(object):

    def __init__(self, 
            cascade_filename=OPENCV_HAAR_CASCADE_FRONTALFACE, 
            scale_factor=OPENCV_SCALE_FACTOR, 
            min_neighbors=OPENCV_MIN_NEIGHBORS, 
            min_size=OPENCV_MIN_SIZE, 
            flags=cv2.cv.CV_HAAR_SCALE_IMAGE, 
            enlarge_factor=ENLARGE_FACTOR):
        # create the haar cascade
        self.cascade = cv2.CascadeClassifier(cascade_filename)

        # remember parameters
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_size = min_size
        self.flags = flags
        self.enlarge_factor = enlarge_factor

    def detect_faces(self, image):
        # convert image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # detect faces in the image
        faces = self.cascade.detectMultiScale(
           gray,
           scaleFactor=self.scale_factor,
           minNeighbors=self.min_neighbors,
           minSize=(self.min_size, self.min_size),
           flags = self.flags
        )

        image_height, image_width, _ = image.shape
        return [enlarge_rectangle(x, y, w, h, image_width, image_height, self.enlarge_factor) for (x, y, w, h) in faces]

def enlarge_rectangle(x, y, w, h, img_w, img_h, factor):
    if factor == 1:
      return x, y, w, h

    # how much we need to add to current width and height
    add_x = int(round((w * factor - w) / 2))
    add_y = int(round((h * factor - h) / 2))

    # fix the added pixels so we don't cross image edges
    add_x = min(add_x, x, img_w - x - w)
    add_y = min(add_y, y, img_h - y - h)

    # TODO: this assumes width and height are equal
    to_add = min(add_x, add_y)

    # enlarge the rectangle
    new_w = w + 2 * to_add
    new_h = h + 2 * to_add
    new_x = x - to_add
    new_y = y - to_add

    return new_x, new_y, new_w, new_h

def scaling_factor(width, height, max_width, max_height):
    '''
    Calculate scaling factor for image with width and height, so that it fits max_width and max_height.
    '''
    scale_x = 1
    scale_y = 1

    if width > max_width:
        scale_x = float(max_width) / width
     
    if height > max_height:
        scale_y = float(max_height) / height

    return min(scale_x, scale_y)

def resize_image_file(image_filename, max_width, max_height):
    '''
    Given a filename, resize the file to have either max_width or max_height. If smaller, then don't resize.
    '''
    image = cv2.imread(image_filename)
    height, width = image.shape[:2]

    scale = scaling_factor(width, height, max_width, max_height)
    if scale != 1:
        width = max(int(width * scale), 1)
        height = max(int(height * scale), 1)
        image = cv2.resize(image, (width, height))

        # write the resized image back to the same file
        cv2.imwrite(image_filename, image)

    return (width, height)

#if __name__ == '__main__':
#    print FaceDetector().detect_faces(sys.argv[1])