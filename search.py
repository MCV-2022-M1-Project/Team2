# Import required packages
from imutils.paths import list_images
from packages import Searcher, RGBHistogram, RemoveBackground
import argparse
import pickle
import collections
import cv2
from packages.average_precicion import mapk
import os

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="Path to the directory that contains the images we just indexed")
# ap.add_argument("-i", "--index", required=False, help="Path to where we stored our index")
ap.add_argument("-q", "--query", required=True, help="Path to query image")
ap.add_argument("-b", "--query1", required=True, help="Path to the query images with background")
ap.add_argument("-t", "--test", required=True, help="Path to the test query images")
ap.add_argument("-t2", "--test2", required=True, help="Path to the test 2 query images")
ap.add_argument("-m", "--masks", required=True, help="Path to save the masks")
ap.add_argument("-mt", "--masks_test", required=True, help="Path to save the test masks")
ap.add_argument("-r1", "--result1", default="output1", help="Path to save the results for the first part")
ap.add_argument("-r2", "--result2", default="output2", help="Path to save the results for the second part")
ap.add_argument("-r3", "--result3", default="output3", help="Path to save the results for the test part 1")
ap.add_argument("-r4", "--result4", default="output4", help="Path to save the results for the test part 2")
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
Results = []

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

# Save the results
with open(args["result1"] + ".pkl", "wb") as fp:
    pickle.dump(predicted, fp)

# Evaluate the map accuracy
print("map@ {}: {}".format(1, evaluate(predicted, args["query"] + "/gt_corresps.pkl", k=1)))

# Initialize the parameters and counters
sumPrecision1 = 0
sumRecall1 = 0
sumF11 = 0
counter1 = 0
sumPrecision2 = 0
sumRecall2 = 0
sumF12 = 0
counter2 = 0

# Initialize predicted and results list
predicted = []
predicted_2 = []
Results = []

# Check if directory exists for the masks
if not os.path.exists(args["masks"]):
    os.mkdir(args["masks"])
if not os.path.exists(args["masks"] + "\\method1"):
    os.mkdir(args["masks"] + "\\method1")
if not os.path.exists(args["masks"] + "\\method2"):
    os.mkdir(args["masks"] + "\\method2")


# Loop over the second dataset with first method
for imagePath in sorted(list_images(args["query1"])):
    mask = cv2.Mat
    mask2 = cv2.Mat

    if "jpg" in imagePath:
        # Get the mask for removing background, load the image
        queryImage = cv2.imread(imagePath)
        mask = RemoveBackground.compute_removal(queryImage)

        # Save the masks for both the methods
        if not cv2.imwrite(args["masks"] + "\\method1" + imagePath[-10:-3] + "png", mask * 255):
            raise Exception("Could not write image")
        mask2 = RemoveBackground.compute_removal_2(queryImage)
        if not cv2.imwrite(args["masks"] + "\\method2" + imagePath[-10:-3] + "png", mask2 * 255):
            raise Exception("Could not write image")

        # Print the query image ame
        print("query: {}".format(imagePath))

        # Describe a 3D RGB histogram with 8 bins per channel
        desc = RGBHistogram((8, 8, 8), mask)
        queryFeatures = desc.compute_histogram(queryImage)

        # Describe a 3D RGB histogram with 8 bins per channel
        desc_2 = RGBHistogram((8, 8, 8), mask2)
        queryFeatures_2 = desc_2.compute_histogram(queryImage)

        # Perform the search
        searcher = Searcher(index)
        results = searcher.search(queryFeatures)
        results_2 = searcher.search(queryFeatures_2)

        # Define the list of predictions
        predicted_query = []
        predicted_query_2 = []

        # Loop over the top 10 results
        print("Method1:")
        for j in range(0, 10):
            # Grab the result
            (score, imageName) = results[j]
            predicted_query.append(int(imageName.replace(".jpg", "")))
            print("\t{}. {} : {:.3f}".format(j + 1, imageName, score))

        # Loop over the top 10 results
        print("Method2:")
        for j in range(0, 10):
            (score_2, imageName_2) = results_2[j]
            predicted_query_2.append(int(imageName_2.replace(".jpg", "")))
            print("\t{}. {} : {:.3f}".format(j + 1, imageName_2, score_2))

        # Append the final predicted list
        predicted.append(predicted_query)
        predicted_2.append(predicted_query_2)

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

        # Add the parameters
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

        # Add the parameters
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

# Save the results in pickle file
with open(args["result2"] + "_1.pkl", "wb") as fp:
    pickle.dump(predicted, fp)

# Save the results in pickle file
with open(args["result2"] + "_2.pkl", "wb") as fp:
    pickle.dump(predicted_2, fp)

# Evaluate the map accuracy
print("Method1: map@ {}: {}".format(1, evaluate(predicted, args["query1"] + "/gt_corresps.pkl", k=1)))
print("Method1: map@ {}: {}".format(5, evaluate(predicted, args["query1"] + "/gt_corresps.pkl", k=5)))

# Evaluate the map accuracy
print("Method2: map@ {}: {}".format(1, evaluate(predicted_2, args["query1"] + "/gt_corresps.pkl", k=1)))
print("Method2: map@ {}: {}".format(5, evaluate(predicted_2, args["query1"] + "/gt_corresps.pkl", k=5)))


# Load the query images in test dataset
for imagePath in sorted(list_images(args["test"])):
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


# Save the results
with open(args["result3"] + ".pkl", "wb") as fp:
    pickle.dump(predicted, fp)


if not os.path.exists(args["masks_test"]):
    os.mkdir(args["masks_test"])
if not os.path.exists(args["masks_test"] + "\\method1"):
    os.mkdir(args["masks_test"] + "\\method1")
if not os.path.exists(args["masks_test"] + "\\method2"):
    os.mkdir(args["masks_test"] + "\\method2")

# Initialize predicted and results list
predicted = []
predicted_2 = []
Results = []

# Loop over the second dataset with first method
for imagePath in sorted(list_images(args["test2"])):
    mask = cv2.Mat
    mask2 = cv2.Mat

    if "jpg" in imagePath:
        # Get the mask for removing background, load the image
        queryImage = cv2.imread(imagePath)
        mask = RemoveBackground.compute_removal(queryImage)

        # Save the masks for both the methods
        if not cv2.imwrite(args["masks_test"] + "\\method1" + imagePath[-10:-3] + "png", mask * 255):
            raise Exception("Could not write image")
        mask2 = RemoveBackground.compute_removal_2(queryImage)
        if not cv2.imwrite(args["masks_test"] + "\\method2" + imagePath[-10:-3] + "png", mask2 * 255):
            raise Exception("Could not write image")

        # Print the query image ame
        print("query: {}".format(imagePath))

        # Describe a 3D RGB histogram with 8 bins per channel
        desc = RGBHistogram((8, 8, 8), mask)
        queryFeatures = desc.compute_histogram(queryImage)

        # Describe a 3D RGB histogram with 8 bins per channel
        desc_2 = RGBHistogram((8, 8, 8), mask2)
        queryFeatures_2 = desc_2.compute_histogram(queryImage)

        # Perform the search
        searcher = Searcher(index)
        results = searcher.search(queryFeatures)
        results_2 = searcher.search(queryFeatures_2)

        # Define the list of predictions
        predicted_query = []
        predicted_query_2 = []

        # Loop over the top 10 results
        print("Method1:")
        for j in range(0, 10):
            # Grab the result
            (score, imageName) = results[j]
            predicted_query.append(int(imageName.replace(".jpg", "")))
            print("\t{}. {} : {:.3f}".format(j + 1, imageName, score))

        # Loop over the top 10 results
        print("Method2:")
        for j in range(0, 10):
            (score_2, imageName_2) = results_2[j]
            predicted_query_2.append(int(imageName_2.replace(".jpg", "")))
            print("\t{}. {} : {:.3f}".format(j + 1, imageName_2, score_2))

        # Append the final predicted list
        predicted.append(predicted_query)
        predicted_2.append(predicted_query_2)

# Save the results in pickle file
with open(args["result4"] + "_1.pkl", "wb") as fp:
    pickle.dump(predicted, fp)

# Save the results in pickle file
with open(args["result4"] + "_2.pkl", "wb") as fp:
    pickle.dump(predicted_2, fp)
