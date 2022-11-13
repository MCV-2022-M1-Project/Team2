# Import required packages
import argparse

import cv2
import imutils
import numpy as np
from imutils import contours, perspective
from imutils.paths import list_images
from packages import extract_angle, RemoveNoise, RemoveBackground
import pickle
from packages import bb_intersection_over_union_rotated

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
    if "jpg" in imagePath1 and "non_augmented" not in imagePath1:

        print(imagePath1)
        image = cv2.imread(imagePath1)
        row, col, _ = image.shape
        
        noise = RemoveNoise(image)

        image = noise.denoise_image()

        th_open, stats = RemoveBackground.compute_removal_2(image)

        angle = extract_angle(th_open)


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

#print("angle_cord_list", angle_cord_list)

file = open("angle_cord_list.pkl", 'rb')
angle_cord_list = pickle.load(file)



def evaluate(predicted, ground_truth):
    file = open(ground_truth, 'rb')
    actual = pickle.load(file)
    IOU = []
    Angular_error = []
    im_count = 0
    for actua_img, predic_img in zip(actual, predicted):
        print("--------")
        print("img:", im_count)
        im_count += 1
        for i in range(len(actua_img)):
            if len(predic_img) > i:
                print("picture:", i)
                iou = bb_intersection_over_union_rotated(actua_img[i][1],predic_img[i][1])
                ang_err = abs(actua_img[i][0] - predic_img[i][0])
                print("iou", iou)
                print("ang_err", ang_err)
                print("")
                IOU.append(iou)
                Angular_error.append(ang_err)
        print("--------")
    return IOU, Angular_error





file = open(args["query1"]+"/frames.pkl", 'rb')
actual = pickle.load(file)

IOU, Angular_error = evaluate(angle_cord_list, args["query1"] + "/frames.pkl")
IOU = np.array(IOU)
print("Mean IOU", IOU.mean())
Angular_error = np.array(Angular_error)
print("Mean angular error", Angular_error.mean())











