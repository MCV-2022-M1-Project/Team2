# Import the necessary packages
import imutils
import numpy as np
from skimage.filters import threshold_local
import argparse
import cv2

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", default="../dataset/qsd2_w2/00000.jpg", help="Path to the image")
args = vars(ap.parse_args())

# Load the image, convert it to grayscale, and blur it 
image = cv2.imread(args["image"])
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(image, (5, 5), 0)
cv2.imshow("Image", image)

# Use OpenCv Adaptive threshold
thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 25, 15)
cv2.imshow("OpenCV Mean Thresh", thresh)

# Use Scikit Learn
T = threshold_local(blurred, 29, offset=5, method="gaussian")
thresh = (blurred < T).astype("uint8") * 255
cv2.imshow("scikit-image Mean Thresh", thresh)
cv2.waitKey(0)

