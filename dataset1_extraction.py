# Evaluate the predictions
import argparse
import collections
import os
import pickle
import cv2
import numpy as np
from imutils.paths import list_images
from packages import HistogramDescriptor, Searcher, TextureDescriptors, TextDescriptors, RemoveNoise
from packages.average_precicion import mapk

def get_k_searcher(index, queryFeatures, k = 10):
    # Perform the search
    searcher = Searcher(index)
    results = searcher.search(queryFeatures)

    predicted_query = []
    # print("text", text)

    # Loop over the top ten results
    for j in range(0, k):
        # Grab the result
        (score, imageName) = results[j]
        predicted_query.append(int(imageName.replace(".jpg", "")))
        print("\t{}. {} : {:.3f}".format(j + 1, imageName, score))

    return predicted_query


def evaluate(predicted, ground_truth, k):
    file = open(ground_truth, 'rb')
    actual = pickle.load(file)
    result = mapk(actual=actual, predicted=predicted, k=k)
    return result


# Construct argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--index", default="../dataset/bbdd", help="Path to the image dataset")
ap.add_argument("-q1", "--query1", default="../dataset/qsd1_w3", help="Path to the query image")
#ap.add_argument("-q2", "--query2", default="../dataset/qsd2_w2", help="Path to the query image")
ap.add_argument("-a", "--augmented", default="n", help="augmented dataset / with noise?")
ap.add_argument("-c", "--color", default="y", help="Do we use color descriptors?")
ap.add_argument("-t", "--texture", default="n", help="Do we use texture descriptors?")
ap.add_argument("-txt", "--text", default="n", help="Do we use text descriptors?")

args = vars(ap.parse_args())

augmented = "y" == args["augmented"]
color = "y" == args["color"]
texture = "y" == args["texture"]
text = "y" == args["text"]


# Initialize a Dictionary to store our images and features
index_color = {}
index_texture = {}
index_text = {}

# Initialize image descriptor
descriptor = HistogramDescriptor((8, 8, 8))
descriptor1 = TextureDescriptors()
print("Indexing images")

if texture or color:
    # Use list_images to grab the image paths and loop over them
    for imagePath in list_images(args["index"]):
        if "jpg" in imagePath:
            # Extract our unique image ID (i.e. the filename)
            path = imagePath[imagePath.rfind("_") + 1:]

            # Load the image, compute histogram and update the index
            image = cv2.imread(imagePath)
            if color:
                index_color[path] = descriptor.computeHSV(image)
            if texture:
                index_texture[path] = descriptor1.compute_hog(image)

if text:
    # Initialize a Dictionary to store texts
    all_files = sorted(os.listdir(args["index"]))
    # Use list_images to grab the image paths and loop over them
    for imagePath in all_files:
        if "txt" in imagePath:
            # Extract our unique image ID (i.e. the filename)
            path = imagePath[imagePath.rfind("_") + 1:]

            # Open the text file and read contents
            file = open(args["index"]+'/'+imagePath, "r")
            line = file.readline()
            if line.strip():
                text = line.lower().replace("(", "").replace("'", " ").replace(")", "")
            else:
                text = 'empty'

            # Add the text to list of dictionaries
            index_text[path] = text

# Sort the dictionary according to the keys
index_color = collections.OrderedDict(sorted(index_color.items()))
index_texture = collections.OrderedDict(sorted(index_texture.items()))
index_text = index_text


text_query = []
predicted_color = []
predicted_texture = []
predicted_text = []
Results = []
bounding_boxes = []

# Load the query images
for imagePath1 in sorted(list_images(args["query1"])):
    if "jpg" in imagePath1:
        if augmented and not "non_augmented" in imagePath1 or not augmented and "non_augmented" in imagePath1:
            queryImage = cv2.imread(imagePath1)
            #queryImage1 = queryImage.copy()
            print("query: {}".format(imagePath1))

            rn = RemoveNoise(queryImage)
            queryImage_rn = rn.denoise_image()
            # Describe a 3D RGB histogram with 8 bins per channel
            if color:
                desc_c = HistogramDescriptor((8, 8, 8))
                queryFeatures_c = desc_c.computeHSV(queryImage) # color works better without removing noise
                predicted_color.append(get_k_searcher(index_color, queryFeatures_c))
            if texture:
                desc_t = TextureDescriptors()
                queryFeatures_t = desc_t.compute_hog(queryImage_rn)
                predicted_texture.append(get_k_searcher(index_texture, queryFeatures_t))
            if text:
                td = TextDescriptors()
                predicted_text.append(td.get_k_images(queryImage_rn, index_text)[0])


# Evaluate the map accuracy
if color:
    print("Prediction of color:")
    print("map@ {}: {}".format(1, evaluate(predicted_color, args["query1"] + "/gt_corresps.pkl", k=1)))
    print("map@ {}: {}".format(5, evaluate(predicted_color, args["query1"] + "/gt_corresps.pkl", k=5)))

if texture:
    print("Prediction of texture:")
    print("map@ {}: {}".format(1, evaluate(predicted_texture, args["query1"] + "/gt_corresps.pkl", k=1)))
    print("map@ {}: {}".format(5, evaluate(predicted_texture, args["query1"] + "/gt_corresps.pkl", k=5)))

if text:
    print("Prediction of text:")
    print("map@ {}: {}".format(1, evaluate(predicted_text, args["query1"] + "/gt_corresps.pkl", k=1)))
    print("map@ {}: {}".format(5, evaluate(predicted_text, args["query1"] + "/gt_corresps.pkl", k=5)))

