# Import required packages
import argparse

import cv2
import imutils
import numpy as np
from imutils import contours, perspective
from imutils.paths import list_images
from packages import extract_angle, RemoveNoise, RemoveBackground
import pickle

# Construct argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--index", default="../dataset/bbdd", help="Path to the image dataset")
ap.add_argument("-q1", "--query1", default="../dataset/qsd1_w5", help="Path to the query image")
args = vars(ap.parse_args())

AnglesList = []
angle_cord_list = []


def order_points_old(pts):
    # Iinitialize a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")

    # The top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # Compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # return the ordered coordinates
    return rect


# Load the query images
for imagePath1 in sorted(list_images(args["query1"])):
    if "jpg" in imagePath1 and "non_augmented" not in imagePath1 and "010.jpg" in imagePath1:

        print(imagePath1)
        image = cv2.imread(imagePath1)
        row, col, _ = image.shape
        
        noise = RemoveNoise(image)

        image = noise.denoise_image()
        angle, image_1 = extract_angle(image)

        th_open, stats = RemoveBackground.compute_removal_2(image)
        cnts = cv2.findContours(th_open, cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        (cnts, _) = contours.sort_contours(cnts)
        angle_photo = []
        for i in range(len(stats)):
            c = cnts[i]
            box = cv2.minAreaRect(c)
            box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
            box = np.array(box, dtype="int")
            rect = order_points_old(box)
            rect = perspective.order_points(box)
            cord_list = rect.astype("int")
            cord_list = [(cord[0],cord[1]) for cord in cord_list]
            angle_photo.append([angle, cord_list])
        angle_cord_list.append(angle_photo)


with open("angle_cord_list" + ".pkl", "wb") as fp:
    pickle.dump(angle_cord_list, fp)
    

print("angle_cord_list", angle_cord_list)
