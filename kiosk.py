import argparse
import numpy as np
import cv2
import os
import sys
from extractor import *
from detector import *
from finder import *

parser = argparse.ArgumentParser()
parser.add_argument("--delay", type=int, default=1)
parser.add_argument("--canvas_width", type=int, default=1024)
parser.add_argument("--canvas_height", type=int, default=768)
parser.add_argument("--face_size", type=int, default=160)
parser.add_argument("--face_count", type=int, default=5)
parser.add_argument("--font_face", type=int, default=cv2.FONT_HERSHEY_SIMPLEX)
parser.add_argument("--font_scale", type=float, default=0.5)
parser.add_argument("--font_thickness", type=int, default=1)
parser.add_argument("--show_name", type=int, default=1)
parser.add_argument("--show_distance", type=int, default=1)
parser.add_argument("--group_by", choices=["file", "name"], default="file")

parser.add_argument("--image_size", type=int, default=64)
parser.add_argument("--face_min_size", type=int, default=100)
parser.add_argument("--min_neighbors", type=int, default=10)
parser.add_argument("--enlarge_factor", type=float, default=1.4)
parser.add_argument("--average_window", type=int, default=10)
parser.add_argument("--oversample", type=int, default=0)
parser.add_argument("--grayscale", type=int, default=1)

parser.add_argument("--images_path", default="../LFW/lfw_resized")
parser.add_argument("--mean_file", default="lfw/lfw+wlf_train_mean.binaryproto")
parser.add_argument("--data_file", default="lfw/lfw_all.csv")
parser.add_argument("--index_file", default="lfw/lfw_all.pkl")
parser.add_argument("--model_file", default="lfw/lfw_features.prototxt")
parser.add_argument("--pretrained_file", default="lfw/lfw+wlf_iter_130000.caffemodel")
'''
parser.add_argument("--images_path", default="../Fotis/isikud_koik_reviewed")
parser.add_argument("--mean_file", default="fotis/lfw+wlf+fotis_train_mean.binaryproto")
parser.add_argument("--data_file", default="fotis/fotis_unlabeled.csv")
parser.add_argument("--index_file", default="fotis/fotis_unlabeled.pkl")
parser.add_argument("--model_file", default="fotis/fotis_features.prototxt")
parser.add_argument("--pretrained_file", default="fotis/lfw+wlf+fotis_iter_110000.caffemodel")
'''
parser.add_argument("--layer", default="ip1")
parser.add_argument("--backend", choices=["gpu", "cpu"], default="cpu")
args = parser.parse_args()

# initialize face detector
detector = FaceDetector(
    min_size=args.face_min_size, 
    min_neighbors=args.min_neighbors, 
    enlarge_factor=args.enlarge_factor, 
    flags=cv2.cv.CV_HAAR_SCALE_IMAGE|cv2.cv.CV_HAAR_FIND_BIGGEST_OBJECT)

# initialize face extractor
extractor = FaceFeaturesExtractor(
    model_snapshot=args.pretrained_file, 
    model_prototxt=args.model_file, 
    mean_file=args.mean_file, 
    image_size=args.image_size, 
    grayscale=(args.grayscale==1),
    oversample=(args.oversample==1),
    layer=args.layer, 
    GPU=(args.backend=="gpu"))

# load nearest neighbor index and metadata
finder = FaceFinder(args.index_file, args.data_file)

# initialize video capture
video = cv2.VideoCapture(0)
frame_width = video.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)
frame_height = video.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)

# initialize canvas and make background white
canvas = np.empty((args.canvas_height, args.canvas_width, 3), dtype=np.uint8)
canvas.fill(255)

# calculate text height
((text_width, text_height), retval) = cv2.getTextSize("Abrakadabra", args.font_face, args.font_scale, args.font_thickness)
assert retval, "Unable to determine text height"

frame_left = int(args.canvas_width / 2 - frame_width / 2)
assert frame_left > 0, "Video frame too big, increase canvas width"
frame_top = int((args.canvas_height - frame_height - args.face_size - text_height) / 3)
assert frame_top > 0, "Video frame too big, increase canvas height"

face_gap = int((args.canvas_width - args.face_count * args.face_size) / (args.face_count + 1))
assert face_gap > 0, "Lower face size"
face_top = int(2 * frame_top + frame_height)

text_top = int(face_top + args.face_size + text_height + 5)

#feature_size = extractor.get_feature_size()
#features = np.empty((args.average_window, feature_size))
results = []
i = 0
while True:
  # capture frame
  ret, frame = video.read()
  assert ret, "Frame could not be read from video capture device"

  # detect faces
  faces = detector.detect_faces(frame)

  # skip following if no faces were found
  if len(faces) == 0:
    canvas[frame_top:frame_top+frame_height, frame_left:frame_left+frame_width, :] = frame
  else:
    # draw rectangle around the face
    frame2 = frame.copy()
    (x, y, w, h) = faces[0]
    cv2.rectangle(frame2, (x, y), (x+w, y+h), (0, 255, 0), 2)
    canvas[frame_top:frame_top+frame_height, frame_left:frame_left+frame_width, :] = frame2

    # extract features of the face
    myface = frame[y:y+h,x:x+w,:]
    #features[i, :] = extractor.get_image_features(myface)
    features = extractor.get_image_features(myface)

    # find nearest images and add to the list
    results += finder.find_nearest_faces(features)

    # after average_window steps count the occurrences
    i = (i + 1) % args.average_window
    if i == 0:
      # find nearest images
      #results = finder.find_nearest_faces(features.mean(0))

      # create dictionary for counting the occurrences
      d = dict()
      for res in results:
        d.setdefault(res[args.group_by], []).append(res)
      d_sorted_keys = sorted(d, key=lambda k: len(d[k]), reverse=True)
      results = []

      # erase previous faces and texts
      canvas[face_top:].fill(255)

      # loop over first face_count faces, that occurred most often
      #for j, res in enumerate(results):
      for j, k in enumerate(d_sorted_keys[:args.face_count]):
        # sort the results for this file/name by distance
        res2 = d[k]
        res3 = sorted(res2, key=lambda r: r["distance"])
        # show the image with smallest distance
        res = res3[0]

        # read the face image
        fullname = os.path.join(args.images_path, res['file'])
        face = cv2.imread(fullname)
        
        # draw the face
        face = cv2.resize(face, (args.face_size, args.face_size))
        face_left = face_gap + j * (face_gap + args.face_size)
        canvas[face_top:face_top+args.face_size,face_left:face_left+args.face_size, :] = face

        if args.show_name:
          # draw the text
          text = res['name']
          ((text_width, text_height), retval) = cv2.getTextSize(text, args.font_face, args.font_scale, args.font_thickness)
          assert retval, "Unable to determine text height"
          text_left = int(face_left + args.face_size / 2 - text_width / 2)
          cv2.putText(canvas, text, (text_left, text_top), args.font_face, args.font_scale, (10, 10, 10), args.font_thickness)

        if args.show_distance:
          text = str(res['distance']) + " (" + str(len(res2)) + ")"
          ((text_width, text_height), retval) = cv2.getTextSize(text, args.font_face, args.font_scale, args.font_thickness)
          assert retval, "Unable to determine text height"
          text_left = int(face_left + args.face_size / 2 - text_width / 2)
          cv2.putText(canvas, text, (text_left, text_top + text_height + 10), args.font_face, args.font_scale, (10, 10, 10), args.font_thickness)

  # display the canvas
  cv2.imshow('Video', canvas)

  key = cv2.waitKey(args.delay)
  if key == 27:
    break;
  elif key == 32:
    print "Saving current images"
    cv2.imwrite("canvas.jpg", canvas)
    cv2.imwrite("frame.jpg", frame)
    cv2.imwrite("myface.jpg", myface)

# when everything is done, release the capture
video.release()
cv2.destroyAllWindows()
