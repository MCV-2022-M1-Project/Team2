# Import required packages
import argparse
import collections
import pickle
import time

import cv2
from imutils.paths import list_images
from imutils.feature.factories import FeatureDetector_create, DescriptorExtractor_create, DescriptorMatcher_create
from packages import HistogramDescriptor, RemoveText, Searcher, RGBHistogram, RemoveBackground, DetectAndDescribe, \
    SearchFeatures
from packages.average_precicion import mapk
import numpy as np
from PIL import Image
import math
from haishoku.haishoku import Haishoku
from skimage.transform import probabilistic_hough_line

# Construct argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--index", default="../dataset/bbdd", help="Path to the image dataset")



pictures = []
palets = []
bins = 2
div = 256/bins

for imagePath in sorted(list_images("../dataset/bbdd")):
    if "jpg" in imagePath:
        print(imagePath)
        original = Image.open(imagePath)
        color = Haishoku.getDominant(imagePath)
        palets.append([int(color[0]/div),int(color[1]/div),int(color[2]/div)]) # get palette of the painting

        
        image = cv2.imread(imagePath)
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.medianBlur(image_gray, 7)
        r,c = blurred.shape

        d_r = 3
        d_c = 3
        r_gap = int(r/d_r)
        c_gap = int(c/d_c)
        sum_dif = 0
        canny = cv2.Canny(blurred, 10, 100)
        center_r = int(d_r/2)
        center_c = int(d_c/2)

        for i in range(d_r):
            for j in range(d_c):
                dif_gap = np.sum(canny[r_gap*center_r:r_gap*(center_r+1), c_gap*center_c:c_gap*(center_c+1)]) - np.sum(canny[r_gap*i:r_gap*(i+1), c_gap*j:c_gap*(j+1)])
                sum_dif += dif_gap > 0


        r_tresh = sum_dif/(r*c) # the bigger the better


        pictures.append((imagePath[-9:-4],r_tresh))


pictures.sort(key = lambda x: x[1], reverse=True)


histo = [[[ [] for _ in range(bins)] for _ in range(bins)] for _ in range(bins)]


for picture in pictures:
    numb_paint = int(picture[0])
    p = palets[numb_paint]
    
    histo[p[0]][p[1]][p[2]].append(picture[0])
    if len(histo[p[0]][p[1]][p[2]]) >= 5:
        break
print(histo[p[0]][p[1]][p[2]])