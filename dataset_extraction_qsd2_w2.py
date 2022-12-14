# Evaluate the predictions
import argparse
import collections
import os
import pickle
import cv2
import numpy as np
from imutils.paths import list_images
from packages import HistogramDescriptor, Searcher, TextureDescriptors, TextDescriptors, RemoveNoise, RemoveBackground
from packages.average_precicion import mapk
from packages.remove_text import RemoveText


def get_k_searcher(index, queryFeatures, k=10):
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


def flatten(l):
    return [[item] for sublist in l for item in sublist]


def evaluate(predicted, ground_truth, k):
    file = open(ground_truth, 'rb')
    actual = pickle.load(file)
    actual = flatten(actual)
    result = mapk(actual=actual, predicted=predicted, k=k)
    return result


# Construct argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--index", default="../dataset/bbdd", help="Path to the image dataset")
ap.add_argument("-q1", "--query1", default="../dataset/qsd2_w2", help="Path to the query image")
# ap.add_argument("-q2", "--query2", default="../dataset/qsd2_w2", help="Path to the query image")
ap.add_argument("-a", "--augmented", default="y", help="augmented dataset / with noise?")
ap.add_argument("-c", "--color", default="y", help="Do we use color descriptors?")
ap.add_argument("-t", "--texture", default="y", help="Do we use texture descriptors?")
ap.add_argument("-txt", "--text", default="y", help="Do we use text descriptors?")

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
            file = open(args["index"] + '/' + imagePath, "r", encoding='latin-1')
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
            # queryImage1 = queryImage.copy()
            print("query: {}".format(imagePath1))

            rn = RemoveNoise(queryImage)
            queryImage_rn = rn.denoise_image()

            th_open, stats = RemoveBackground.compute_removal_2(queryImage_rn)
            cv2.imwrite("masks/" + imagePath + ".png", th_open)

            pq = []
            bgb = []

            for i in range(0, len(stats)):
                bb = stats[i]
                image1 = queryImage.copy()

                text_id = RemoveText(queryImage_rn[bb[1]:bb[1] + bb[4], bb[0]:bb[0] + bb[2], :])
                bbox = text_id.extract_text()
                image_rec = cv2.rectangle(queryImage_rn[bb[1]:bb[1] + bb[4], bb[0]:bb[0] + bb[2], :],
                                          (bbox[0], bbox[1]),
                                          (bbox[2], bbox[3]), (0, 0, 255), 8)

                # Describe a 3D RGB histogram with 8 bins per channel
                if color:
                    desc_c = HistogramDescriptor((8, 8, 8))
                    queryFeatures_c = desc_c.computeHSV(queryImage[bb[1]:bb[1] + bb[4], bb[0]:bb[0] + bb[2],
                                                        :])  # color works better without removing noise
                    predicted_color.append(get_k_searcher(index_color, queryFeatures_c))
                if texture:
                    desc_t = TextureDescriptors()
                    queryFeatures_t = desc_t.compute_hog(queryImage_rn[bb[1]:bb[1] + bb[4], bb[0]:bb[0] + bb[2], :])
                    predicted_texture.append(get_k_searcher(index_texture, queryFeatures_t))
                if text:
                    td = TextDescriptors()
                    predicted_text.append(td.get_k_images(queryImage_rn, index_text)[0])


with open("output_color3" + ".pkl", "wb") as fp:
    pickle.dump(predicted_color, fp)

with open("output_texture3" + ".pkl", "wb") as fp:
    pickle.dump(predicted_texture, fp)

with open("output_text3" + ".pkl", "wb") as fp:
    pickle.dump(predicted_text, fp)

file = open("output_color3.pkl", 'rb')
predicted_color = pickle.load(file)
file = open("output_texture3.pkl", 'rb')
predicted_texture = pickle.load(file)
file = open("output_text3.pkl", 'rb')
predicted_text = pickle.load(file)

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


def calculate_one_descriptor(voting, predicted, prob):
    for i, l in enumerate(predicted):
        v = voting[i]
        for j, p in enumerate(l):

            if not p in v:
                v[p] = prob * (len(predicted) - j)
            else:
                v[p] = v[p] + prob * (len(predicted) - j)


def calculate_soft_voting(p_color, p_texture, p_text):
    voting = []
    longi = max(len(p_color), len(p_texture), len(p_text))
    for it in range(longi):
        voting.append({})
    calculate_one_descriptor(voting, p_color, 0.8)
    calculate_one_descriptor(voting, p_texture, 0.733)
    calculate_one_descriptor(voting, p_text, 0.266)

    voted = [sorted(v.items(), key=lambda item: item[1], reverse=True) for v in voting]

    return [[p[0] for p in predictions] for predictions in voted]


predicted_all = calculate_soft_voting(predicted_color, predicted_texture, predicted_text)

print("Prediction of a combination of all:")
print("map@ {}: {}".format(1, evaluate(predicted_all, args["query1"] + "/gt_corresps.pkl", k=1)))
print("map@ {}: {}".format(5, evaluate(predicted_all, args["query1"] + "/gt_corresps.pkl", k=5)))
