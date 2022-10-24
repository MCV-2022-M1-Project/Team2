# Evaluate the predictions
import argparse
import collections
import pickle
import cv2
import numpy as np
from imutils.paths import list_images
from packages import HistogramDescriptor, RemoveText, Searcher, TextureDescriptors, read_text, get_text_distance, get_k_images
from packages.average_precicion import mapk


def evaluate(predicted, ground_truth, k):
    file = open(ground_truth, 'rb')
    actual = pickle.load(file)
    result = mapk(actual=actual, predicted=predicted, k=k)
    return result


# Construct argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--index", default="../dataset/bbdd", help="Path to the image dataset")
ap.add_argument("-q1", "--query1", default="../dataset/qsd1_w3", help="Path to the query image")
ap.add_argument("-q2", "--query2", default="../dataset/qsd2_w2", help="Path to the query image")
args = vars(ap.parse_args())

# Initialize a Dictionary to store our images and features
index = {}

# Initialize image descriptor
descriptor = HistogramDescriptor((8, 8, 8))
descriptor1 = TextureDescriptors()
print("Indexing images")

# Use list_images to grab the image paths and loop over them
for imagePath in list_images(args["index"]):
    if "jpg" in imagePath:
        # Extract our unique image ID (i.e. the filename)
        path = imagePath[imagePath.rfind("_") + 1:]

        # Load the image, compute histogram and update the index
        image = cv2.imread(imagePath)
        features = np.concatenate([descriptor.computeHSV(image), descriptor1.compute_hog(image)])
        index[path] = features

# Initialize a Dictionary to store texts
index_text = {}

# Use list_images to grab the image paths and loop over them
for imagePath in list_images(args["index"]):
    if "txt" in imagePath:
        # Extract our unique image ID (i.e. the filename)
        path = imagePath[imagePath.rfind("_") + 1:]

        # Open the text file and read contents
        file = open(imagePath, "r")
        line = file.readline()
        if line.strip():
            text = line.lower().replace("(", "").replace("'", " ").replace(")", "")
        else:
            text = 'empty'

        # Add the text to list of dictionaries
        index_text[path] = text

# Sort the dictionary according to the keys
index = collections.OrderedDict(sorted(index.items()))
index_text = collections.OrderedDict(sorted(index_text.items()))
text_query = []
predicted = []
Results = []
bounding_boxes = []

# Load the query images
for imagePath1 in sorted(list_images(args["query1"])):
    if "jpg" in imagePath1:
        queryImage = cv2.imread(imagePath1)
        queryImage1 = queryImage.copy()
        print("query: {}".format(imagePath1))

        text_id = RemoveText(queryImage)
        bbox = text_id.extract_text()
        text = read_text(queryImage, bbox)
        text_query.append(text)
        image = cv2.rectangle(queryImage1, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 4)
        # cv2.imshow("image", image)
        # cv2.waitKey(0)
        # mean, std = cv2.meanStdDev(queryImage)
        # mean = [int(val) for val in mean]
        # queryImage[bbox[1]:bbox[3], bbox[0]:bbox[2]] = mean

        # Describe a 3D RGB histogram with 8 bins per channel
        desc = HistogramDescriptor((8, 8, 8))
        desc1 = TextureDescriptors()
        queryFeatures = np.concatenate([desc.computeHSV(image), desc1.compute_hog(image)])

        # Perform the search
        searcher = Searcher(index)
        results = searcher.search(queryFeatures)
        # text_results = get_k_images(text=text, index=index_text)
        predicted_query = []
        # print("text", text)

        # Loop over the top ten results
        for j in range(0, 10):
            # Grab the result
            (score, imageName) = results[j]
            # (score_text, imageName_text) = text_results[j]
            predicted_query.append(int(imageName.replace(".jpg", "")))
            print("\t{}. {} : {:.3f}".format(j + 1, imageName, score))
            #print("text")
            # print("\t{}. {} : {:.3f}".format(j + 1, imageName_text, score_text))

        # Append the final predicted list
        predicted.append(predicted_query)
        # bounding_boxes.append(bbox)

# Evaluate the map accuracy
print("map@ {}: {}".format(1, evaluate(predicted, args["query1"] + "/gt_corresps.pkl", k=1)))
print("map@ {}: {}".format(5, evaluate(predicted, args["query1"] + "/gt_corresps.pkl", k=5)))

# Save the results
with open("output_2" + ".pkl", "wb") as fp:
    pickle.dump(predicted, fp)
with open("bounding_boxes_2" + ".pkl", "wb") as fp:
    pickle.dump(bounding_boxes, fp)
