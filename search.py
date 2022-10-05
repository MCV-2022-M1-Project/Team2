# Import required packages
from packages import Searcher
from packages import RGBHistogram
import numpy as np
import argparse
import os
import pickle
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="Path to the directory that contains the images we just indexed")
ap.add_argument("-i", "--index", required=True, help="Path to where we stored our index")
ap.add_argument("-q", "--query", required=True, help="Path to query image")
args = vars(ap.parse_args())

# load the query image and show it
queryImage = cv2.imread(args["query"])
cv2.imshow("Query", queryImage)
print("query: {}".format(args["query"]))

# describe the query in the same way that we did in
# index.py -- a 3D RGB histogram with 8 bins per channel
desc = RGBHistogram([8, 8, 8])
queryFeatures = desc.compute_histogram(queryImage)

# load the index perform the search
index = pickle.loads(open(args["index"], "rb").read())
searcher = Searcher(index)
results = searcher.search(queryFeatures)


# loop over the top ten results
for j in range(0, 10):
    # grab the result (we are using row-major order) and
    # load the result image
    (score, imageName) = results[j]
    path = os.path.join(args["dataset"], "bbdd_" + imageName)
    result = cv2.imread(path)
    print("\t{}. {} : {:.3f}".format(j + 1, imageName, score))
