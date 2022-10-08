# Import required packages
from imutils.paths import list_images
from packages import Searcher
from packages import RGBHistogram
import argparse
import pickle
import collections
import cv2
from packages import RemoveBackground
from packages.average_precicion import mapk
import os

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="Path to the directory that contains the images we just indexed")
ap.add_argument("-i", "--index", required=False, help="Path to where we stored our index")
ap.add_argument("-q", "--query", required=True, help="Path to query image")
ap.add_argument("-b", "--query1", required=True, help="Path to the query images with background")
ap.add_argument("-m", "--masks", required=True, help="Path to save the masks")
args = vars(ap.parse_args())


# Evaluate the predictions
def evaluate(predicted, ground_truth, k):
    file = open(ground_truth, 'rb')
    actual = pickle.load(file)
    result = mapk(actual=actual, predicted=predicted, k=k)
    return result


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
predicted = []

# Load the query images
for imagePath in sorted(list_images(args["query"])):
    if "jpg" in imagePath:
        queryImage = cv2.imread(imagePath)
        print("query: {}".format(imagePath))

        # Describe a 3D RGB histogram with 8 bins per channel
        desc = RGBHistogram((8, 8, 8), None)
        queryFeatures = desc.compute_histogram(queryImage)

        # Perform the search
        searcher = Searcher(index)
        results = searcher.search(queryFeatures)
        predicted_query = []

        # Loop over the top ten results
        for j in range(0, 10):
            # Grab the result
            (score, imageName) = results[j]
            predicted_query.append(int(imageName.replace(".jpg", "")))
            print("\t{}. {} : {:.3f}".format(j + 1, imageName, score))

        # Append the final predicted list
        predicted.append(predicted_query)

# Evaluate the map accuracy
print("map@ {}: {}".format(5, evaluate(predicted, args["query"] + "/gt_corresps.pkl", k=5)))

# Initialize the parameters
sumPrecision1 = 0
sumRecall1 = 0
sumF11 = 0
counter1 = 0
sumPrecision2 = 0
sumRecall2 = 0
sumF12 = 0
counter2 = 0
predicted = []
if not os.path.exists(args["masks"] + "\\Method1"):
    os.mkdir(args["masks"] + "\\Method1")
if not os.path.exists(args["masks"] + "\\Method2"):
    os.mkdir(args["masks"] + "\\Method2")

# Loop over the second dataset with first method
for imagePath in sorted(list_images(args["query1"])):
    mask = cv2.Mat
    mask2 = cv2.Mat
    if "jpg" in imagePath:
        # Get the mask for removing background, load the image
        queryImage = cv2.imread(imagePath)
        mask = RemoveBackground.compute_removal(queryImage)
        if not cv2.imwrite(args["masks"] + "\\Method1" + imagePath[-10:-3] + "png", mask * 255):
            raise Exception("Could not write image")
        mask2 = RemoveBackground.compute_removal_2(queryImage)
        if not cv2.imwrite(args["masks"] + "\\Method2" + imagePath[-10:-3] + "png", mask2 * 255):
            raise Exception("Could not write image")
        queryImage = cv2.bitwise_and(queryImage, queryImage, mask=mask)
        print("query: {}".format(imagePath))

        # Describe a 3D RGB histogram with 8 bins per channel
        desc = RGBHistogram((8, 8, 8), None)
        queryFeatures = desc.compute_histogram(queryImage)

        # Perform the search
        searcher = Searcher(index)
        results = searcher.search(queryFeatures)
        predicted_query = []

        # Loop over the top ten results
        for j in range(0, 10):
            # Grab the result
            (score, imageName) = results[j]
            predicted_query.append(int(imageName.replace(".jpg", "")))
            print("\t{}. {} : {:.3f}".format(j + 1, imageName, score))

        # Append the final predicted list
        predicted.append(predicted_query)

        # Save the path to the mask and get directions to original mask
        ogMask = cv2.imread(imagePath[:-3] + "png")
        height, width, _ = ogMask.shape

        # Initialize the precision parameters
        tp1 = 0
        fp1 = 0
        fn1 = 0

        # Loop over the original mask
        for i in range(height):
            for j in range(width):
                if ogMask[i, j, 0] == 0 and mask[i, j] != 0:
                    fp1 += 1
                elif ogMask[i, j, 0] != 0 and mask[i, j] == 0:
                    fn1 += 1
                elif ogMask[i, j, 0] != 0 and mask[i, j] != 0:
                    tp1 += 1

        # Calculate the parameters
        precision1 = tp1 / (tp1 + fp1)
        recall1 = tp1 / (tp1 + fn1)
        f11 = 2 * precision1 * recall1 / (precision1 + recall1)

        # Add the paramters
        sumPrecision1 += precision1
        sumRecall1 += recall1
        sumF11 += f11
        counter1 += 1

        # Take the average
        avgPrecision1 = sumPrecision1 / counter1
        avgRecall1 = sumRecall1 / counter1
        avgF11 = sumF11 / counter1

        # Print the values
        print("Method 1 Precision: ", avgPrecision1 * 100, "%")
        print("Method 1 Recall: ", avgRecall1 * 100, "%")
        print("Method 1 F1: ", avgF11 * 100, "%")

        # Initialize the precision parameters
        tp2 = 0
        fp2 = 0
        fn2 = 0

        # Loop over the original mask
        for i in range(height):
            for j in range(width):
                if ogMask[i, j, 0] == 0 and mask2[i, j] != 0:
                    fp2 += 1
                elif ogMask[i, j, 0] != 0 and mask2[i, j] == 0:
                    fn2 += 1
                elif ogMask[i, j, 0] != 0 and mask2[i, j] != 0:
                    tp2 += 1

        # Calculate the parameters
        precision2 = tp2 / (tp2 + fp2)
        recall2 = tp2 / (tp2 + fn2)
        f12 = 2 * precision2 * recall2 / (precision2 + recall2)

        # Add the paramters
        sumPrecision2 += precision2
        sumRecall2 += recall2
        sumF12 += f12
        counter2 += 1

        # Take the average
        avgPrecision2 = sumPrecision2 / counter2
        avgRecall2 = sumRecall2 / counter2
        avgF12 = sumF12 / counter2

        # Print the values
        print("Method 2 Precision: ", avgPrecision2 * 100, "%")
        print("Method 2 Recall: ", avgRecall2 * 100, "%")
        print("Method 2 F1: ", avgF12 * 100, "%\n")

        

# Evaluate the map accuracy
print("map@ {}: {}".format(5, evaluate(predicted, args["query1"] + "/gt_corresps.pkl", k=5)))
