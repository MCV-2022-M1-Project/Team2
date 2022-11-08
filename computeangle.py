# Import required packages
import argparse

import cv2
from imutils.paths import list_images

from packages import extract_angle

# Construct argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--index", default="../dataset/bbdd", help="Path to the image dataset")
ap.add_argument("-q1", "--query1", default="../dataset/qsd1_w5", help="Path to the query image")
ap.add_argument("-q2", "--query2", default="../dataset/qsd2_w2", help="Path to the query image")
ap.add_argument("-m", "--matcher", type=str, default="BruteForce",
                help="Feature matcher to use. Options ['BruteForce', 'BruteForce-SL2', 'BruteForce-L1', 'FlannBased']")

args = vars(ap.parse_args())
AnglesList = []

# Load the query images
for imagePath1 in sorted(list_images(args["query1"])):
    if "jpg" in imagePath1 and "non_augmented" not in imagePath1:
        print(imagePath1)
        image = cv2.imread(imagePath1)
        ang = []

        # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        angle1 = extract_angle(image)
        print(angle1)
        ang.append(angle1)

        # Append the final predicted list
        AnglesList.append(ang)
        # print(predicted)

print(AnglesList)
