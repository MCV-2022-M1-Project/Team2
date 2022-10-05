from packages import RGBHistogram
import argparse

import matplotlib.pyplot as plt
import cv2
import numpy as np
import matplotlib.image as mpimg
from PIL import Image
from statistics import median
threshold = 60
threshold_2 = 20
threshold_rec = 350


"""
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-q", "--query", required=True, help="Path to query image")
ap.add_argument("-o", "--output", required=True, help="Path to where we stored our index")
args = vars(ap.parse_args())
"""
name = './qsd2_w1/00022.jpg'
imagecv = cv2.imread(name)
image = Image.open(name)
im = image.convert('RGBA')
data = np.array(im)

dataR = data[:,:,0]
dataG = data[:,:,1]
dataB = data[:,:,2]


fil, col, _ = data.shape

valor = data[0, round(col/2), :].astype(int)
valor = valor + data[fil-1, round(col/2), :].astype(int)
valor = valor + data[round(fil/2), 0 , :].astype(int)
valor = valor + data[round(fil/2), col-1, :].astype(int)
valor = valor / 4


mask = np.logical_not((data[:,:,0] > valor[0] - threshold) & (data[:,:,0] < valor[0]  + threshold) & (data[:,:,1] > valor[1]  - threshold) & (data[:,:,1] < valor[1]  + threshold) & (data[:,:,2] > valor[2]  - threshold) & (data[:,:,2] < valor[2] + threshold))

for i in range(fil):
    first = 0
    final = col
    found_first = False

    for j in range(col):
        if mask[i,j] > 0 and not found_first:
            found_first = True
            first = j
        if mask[i,j] > 0 and j - final < threshold_rec:
            final = j
    if found_first:
        mask[i,first:final] = 1
    

for j in range(col):
    first = 0
    final = fil
    found_first = False

    for i in range(fil):
        if mask[i,j] > 0 and not found_first:
            found_first = True
            first = i
        if mask[i,j] > 0 and i - final < threshold_rec:
            final = i
    if found_first:
        mask[first:final,j] = 1


mask = mask.astype("uint8")

masked = cv2.bitwise_and(imagecv, imagecv, mask=mask)

cv2.imwrite("mascara.png", masked)
