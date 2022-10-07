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
sumPrecision = 0
sumRecall = 0
sumF1 = 0
counter = 0
predicted = []

# Loop over the second dataset
for imagePath in sorted(list_images(args["query1"])):
    mask = cv2.Mat
    if "jpg" in imagePath:
        # Get the mask for removing background, load the image
        queryImage = cv2.imread(imagePath)
        mask = RemoveBackground.compute_removal(queryImage)
        pth = args["masks"] + imagePath[-10:-3] + "png"
        print(imagePath, pth)
        if not cv2.imwrite(pth, mask * 255):
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
        maskPath = imagePath[:-3] + "png"
        ogMask = cv2.imread(maskPath)
        height, width, _ = ogMask.shape

        # Initialize the precision parameters
        tp = 0
        tn = 0
        fp = 0
        fn = 0

        # Loop over the original mask
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

        # Calculate the parameters
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * precision * recall / (precision + recall)

        # Add the paramters
        sumPrecision += precision
        sumRecall += recall
        sumF1 += f1
        counter += 1

        # Take the average
        avgPrecision = sumPrecision / counter
        avgRecall = sumRecall / counter
        avgF1 = sumF1 / counter

        # Print the values
        print("Precision: ", avgPrecision * 100, "%")
        print("Recall: ", avgRecall * 100, "%")
        print("F1: ", avgF1 * 100, "%\n")

# Evaluate the map accuracy
print("map@ {}: {}".format(5, evaluate(predicted, args["query1"] + "/gt_corresps.pkl", k=5)))
