# Import required packages
import cv2
from packages import RemoveNoise, RemoveText, TextDescriptors
from imutils.paths import list_images
import pickle
from packages.average_precicion import mapk
import os
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--index", default="../dataset/bbdd", help="Path to the image dataset")
ap.add_argument("-q1", "--query1", default="../dataset/qsd1_w3", help="Path to the query image")
args = vars(ap.parse_args())

index_text = {}

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



predicted = []
for imagePath in list_images(args["query1"]):
    if "jpg" in imagePath and not "non_augmented" in imagePath:
        print(imagePath)
        image = cv2.imread(imagePath)
        rm = RemoveNoise(image)
        image = rm.denoise_image()

        td = TextDescriptors()
        predicted.append(td.get_k_images(image, index_text)[0])

# Evaluate the predictions
def evaluate(predicted, ground_truth, k):
    file = open(ground_truth, 'rb')
    actual = pickle.load(file)

    result = mapk(actual=actual, predicted=predicted, k=k)
    return result

print("map@ {}: {}".format(1, evaluate(predicted, args["query1"] + "/gt_corresps.pkl", k=1)))
print("map@ {}: {}".format(5, evaluate(predicted, args["query1"] + "/gt_corresps.pkl", k=5)))


