#!/usr/bin/env python

import sys
import numpy as np
from sklearn.neighbors import NearestNeighbors
import cPickle as pickle

if len(sys.argv) < 3:
  print "Usage:", sys.argv[0], "<features_npz> <pickle_file>"
  sys.exit(2)

features_npz = sys.argv[1]
pickle_file = sys.argv[2]

print "Loading data..."
features = np.load(features_npz)

print "Training model..."
nn = NearestNeighbors(n_neighbors=5)
nn.fit(features['data'])

print "Saving model..."
pickle.dump(nn, open(pickle_file, "wb"), protocol=pickle.HIGHEST_PROTOCOL)

print "Done"