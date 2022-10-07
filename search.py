# Import required packages
#from curses import tparm
from imutils.paths import list_images
from packages import Searcher
from packages import RGBHistogram
import argparse
import os
import pickle
import collections
import cv2
from packages import RemoveBackground

# from metrics.average_precision import mapk

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="Path to the directory that contains the images we just indexed")
ap.add_argument("-i", "--index", required=False, help="Path to where we stored our index")
ap.add_argument("-q", "--query", required=True, help="Path to query image")
ap.add_argument("-b", "--query1", required=True, help="Path to the query images with background")
args = vars(ap.parse_args())
#args = {}
#args["dataset"] = "BBDD"
#args["index"] = "."
#args["query"] = "qsd1_w1"
#args["query1"] = "qsd2_w1"

# Initialize a Dictionary to store our images and features
index = {}

# Initialize image descriptor
descriptor = RGBHistogram((8, 8, 8), None)
print("Indexing images")

# Use list_images to grab the image paths and loop over them
for imagePath in list_images(args["dataset"]):
    if "jpg" in imagePath:
        # Extract our unique image ID (i.e. the filename)
        path = imagePath[imagePath.rfind("_") + 1:]

        # Load the image, compute histogram and update the index
        image = cv2.imread(imagePath)
        features = descriptor.compute_histogram(image)
        index[path] = features

# Sort the dictionary according to the keys
index = collections.OrderedDict(sorted(index.items()))

# load the query image and show it
for imagePath in sorted(list_images(args["query"])):
    if "jpg" in imagePath:
        queryImage = cv2.imread(imagePath)
        #cv2.imshow("Query", queryImage)
        print("query: {}".format(imagePath))

        # describe the query in the same way that we did in
        # index.py -- a 3D RGB histogram with 8 bins per channel
        desc = RGBHistogram((8, 8, 8), None)
        queryFeatures = desc.compute_histogram(queryImage)

        # load the index perform the search
        searcher = Searcher(index)
        results = searcher.search(queryFeatures)

        # loop over the top ten results
        for j in range(0, 10):
            # grab the result (we are using row-major order) and
            # load the result image
            (score, imageName) = results[j]
            print("\t{}. {} : {:.3f}".format(j + 1, imageName, score))

# load the query image and show it
sumPrecision = 0
sumRecall = 0
sumF1 = 0
counter = 0
for imagePath in sorted(list_images(args["query1"])):

    mask = cv2.Mat
    if "jpg" in imagePath:
        # Get the mask for removing background, load the image
        queryImage = cv2.imread(imagePath)
        mask = RemoveBackground.compute_removal(queryImage)
        queryImage = cv2.bitwise_and(queryImage, queryImage, mask=mask)
        print("query: {}".format(imagePath))

        # describe the query in the same way that we did in
        # index.py -- a 3D RGB histogram with 8 bins per channel
        desc = RGBHistogram((8, 8, 8), None)
        queryFeatures = desc.compute_histogram(queryImage)

        # load the index perform the search
        searcher = Searcher(index)
        results = searcher.search(queryFeatures)

        # loop over the top ten results
        for j in range(0, 10):
            # grab the result (we are using row-major order) and
            # load the result image
            (score, imageName) = results[j]
            print("\t{}. {} : {:.3f}".format(j + 1, imageName, score))

        maskPath = imagePath[:-3] + "png"
        ogMask = cv2.imread(maskPath)
        height, width, _ = ogMask.shape
        tp = 0
        tn = 0
        fp = 0
        fn = 0

        for i in range(height):
            for j in range(width):
                if ogMask[i, j, 0] == 0 and mask[i, j] == 0:
                    tn += 1
                elif ogMask[i, j, 0] == 0 and mask[i, j] != 0:
                    fp += 1 
                elif ogMask[i, j, 0] != 0 and mask[i, j] == 0:
                    fn += 1
                elif ogMask[i, j, 0] != 0 and mask[i, j] != 0:
                    tp += 1
                #if ogMask[i, j, 0] == 0 and ogMask[i, j, 0] == 0:
                #    tn += 1
                #elif ogMask[i, j, 0] == 0 and ogMask[i, j, 0] != 0:
                #    fp += 1 
                #elif ogMask[i, j, 0] != 0 and ogMask[i, j, 0] == 0:
                #    fn += 1
                #elif ogMask[i, j, 0] != 0 and ogMask[i, j, 0] != 0:
                #    tp += 1
        
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * precision * recall / (precision + recall)

        sumPrecision += precision
        sumRecall += recall
        sumF1 += f1
        counter += 1

        avgPrecision = sumPrecision / counter
        avgRecall = sumRecall / counter
        avgF1 = sumF1 / counter

        print("Precision: ", avgPrecision * 100, "%")
        print("Recall: ", avgRecall * 100, "%")
        print("F1: ", avgF1 * 100, "%\n")
