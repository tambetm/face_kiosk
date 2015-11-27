'''
facedbsearch: Main module

Copyright 2015, Tambet Matiisen
Licensed under MIT.
'''
from sklearn.neighbors import NearestNeighbors
import cPickle as pickle
import csv

class FaceFinder(object):
    def __init__(self, index_filename, data_filename):
        self.index_filename = index_filename
        self.nn = pickle.load(open(index_filename, "rb"))

        # load metadata
        with open(data_filename, 'rb') as f:
            reader = csv.reader(f)
            self.data = [row for row in reader]

    def find_nearest_faces(self, feature_vector):
        dists, classes = self.nn.kneighbors(feature_vector, return_distance=True)
        dists, classes = (dists[0], classes[0])
        results = [{'file': self.data[n][1], 'distance': dists[i], 'description': self.data[n][2]} for i, n in enumerate(classes)]
        return results
