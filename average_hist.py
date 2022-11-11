# Import required packages
import argparse
import cv2
from imutils.paths import list_images
from matplotlib import pyplot as plt
import numpy as np

# Construct argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--index", default="../dataset/bbdd", help="Path to the image dataset")




sv = []
contours = []

for imagePath in sorted(list_images("../dataset/BBDD")):
    if "jpg" in imagePath:
        print(imagePath)

        image = cv2.imread(imagePath)
        hls = cv2.cvtColor(image,cv2.COLOR_BGR2HLS)

        r,c = hls[:,:,1].shape
        l = np.sum(hls[:,:,1])
        s = np.sum(hls[:,:,2])

        count = r*c

        sv.append(1/(l / count)+1/(s / count)) # lower l and s better

        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.medianBlur(image_gray, 7)
        r,c = blurred.shape
        canny = cv2.Canny(blurred, 10, 100)
        thresh = np.sum(canny > 0)
        r_tresh = (thresh/(r*c)) # bigger thresh better


        contours.append(r_tresh)

sv = [float(i)/max(sv) for i in sv]
contours = [float(i)/max(contours) for i in contours]


best = []
for i in range(len(sv)):
    mult_sv = 0.5
    mult_cont = 0.5
    score = mult_sv * sv[i] + mult_cont * (1-abs(contours[i]-0.6))
    best.append(((str(i).zfill(5)), score))

best.sort(key = lambda x: x[1], reverse=True)


print(best[:5])