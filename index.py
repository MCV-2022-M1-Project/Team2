# Import required packages
import csv
import collections
from packages import RGBHistogram
from packages import HistogramDescriptor
from imutils.paths import list_images
import argparse
import pickle
import cv2

# Construct argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="Path to the directory that contains the images to be indexed")
ap.add_argument("-i", "--index", required=True, help="Path to where the computed index will be stored")
ap.add_argument("-p", "--index1", required=True, help="Path to where the computed index will be stored")
args = vars(ap.parse_args())

# Initialize a Dictionary to store our images and features
index = {}

# Initialize image descriptor
descriptor = HistogramDescriptor([8, 8, 8])
print("Indexing images")

# Use list_images to grab the image paths and loop over them
for imagePath in list_images(args["dataset"]):
    if "jpg" in imagePath:
        # extract our unique image ID (i.e. the filename)
        path = imagePath[imagePath.rfind("_") + 1:]

        # Load the image, compute histogram and update the index
        image = cv2.imread(imagePath)
        features = descriptor.computeHSV(image)
        index[path] = features

# Sort the dictionary according to the keys
index = collections.OrderedDict(sorted(index.items()))

# Write index to disk
f = open(args["index"], "wb")
f.write(pickle.dumps(index))
f.close()

f1 = open(args["index1"], "wb")
f1.write(pickle.dumps(index))
f1.close()

# open file for writing
w = csv.writer(open("output_query.csv", "w"))

# loop over dictionary keys and values
for key, val in index.items():
    # write every key and value to file
    w.writerow([key, val])

# Show how many images we indexed
print("Indexed {} Images".format(len(index)))
