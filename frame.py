
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
from collections import deque

# Construct argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--index", default="../dataset/bbdd", help="Path to the image dataset")

def BFS(thresh, bb):
        queue = []
        row, col = thresh.shape
        mask = np.ones((row,col)).astype("uint8")*255
        mask[bb[0],bb[1]] = 0
        queue = deque([(bb[0],bb[1])])
        xmax = 0
        xmin = row
        ymax = 0
        ymin = col

        while len(queue) > 0:
            i,j = queue[0]
            if i > xmax:
                xmax = i
            if i < xmin:
                xmin = i
            if j > ymax:
                ymax = j
            if j < ymin:
                ymin = j
            queue.popleft()


            if i-1 >= 0 and thresh[i-1, j] <= 0:
                if (i-1,j) not in queue and mask[i-1,j] > 0: 
                    queue.append((i-1,j))
                    mask[i-1,j] = 0

            if i+1 < row and thresh[i+1, j] <= 0:
                if (i+1,j) not in queue and mask[i+1,j] > 0:
                    queue.append((i+1,j))
                    mask[i+1,j] = 0

            if j-1 >= 0 and thresh[i, j-1] <= 0:
                if (i,j-1) not in queue and mask[i,j-1] > 0:
                    queue.append((i,j-1))
                    mask[i,j-1] = 0

            if j+1 < col and thresh[i, j+1] <= 0:
                if (i,j+1) not in queue and mask[i,j+1] > 0:
                    queue.append((i,j+1))
                    mask[i,j+1] = 0
            
        return mask, (xmin,xmax,ymin,ymax)

pictures = []
palets = []
bins = 2
div = 256/bins

for imagePath in sorted(list_images("../dataset/bbdd")):
    if "jpg" in imagePath:
        print(imagePath)
        original = Image.open(imagePath) # open RGB image
        color = Haishoku.getDominant(imagePath)
        palets.append([int(color[0]/div),int(color[1]/div),int(color[2]/div)])

       

        image = cv2.imread(imagePath)
        image = cv2.resize(image, (400,400), interpolation = cv2.INTER_LANCZOS4)
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        blurred = cv2.medianBlur(image_gray, 7)
        r,c = blurred.shape
        canny = cv2.Canny(blurred, 10, 100)
        #cv2.imwrite("canny.png",canny)

        #lines = probabilistic_hough_line(canny, threshold=300, line_length=int(np.floor(np.size(image, 0) / 5)), line_gap=3)
        lines_2 = cv2.HoughLinesP(canny, 1, math.pi / 180.0, 100, minLineLength=100, maxLineGap=r*c/100)
        

        lines = image_gray * 0
        th_r = r/5
        th_c = c/5
        if lines_2 is not None:
            for x in range(0, len(lines_2)):
                for x1,y1,x2,y2 in lines_2[x]:
                    if abs(x1 - x2) < 5 or abs(y1 - y2) < 5:
                        if not ((x1 < (r/2 + th_r) and (x1 > (r/2 - th_r))) or (y1 < (c/2 + th_c) and (y1 > (c/2 - th_c)))) :
                            lines = cv2.line(lines,(x1,y1),(x2,y2),(255,255,255),9)
            #cv2.imwrite("canny_lines.png",lines)

            mask, (xmin,xmax,ymin,ymax) = BFS(lines,[int(r/2),int(c/2)])

            #cv2.imwrite("mask.png",mask)
            
            
            img_mask = cv2.bitwise_and(image_gray, mask)
            #cv2.imwrite("canny_image_bbb.png", img_mask)
            th_ratio = 0.1
            th_r_center = r/20
            th_c_center = c/20 
            #print((xmax-xmin)/(ymax-ymin))
            if (xmax-xmin)/(ymax-ymin) > (r/c - th_ratio) and (xmax-xmin)/(ymax-ymin) < (r/c + th_ratio) and ((xmax-xmin)/2 + xmin > r/2 - th_r_center and (xmax-xmin)/2 + xmin < r/2 + th_r_center and (ymax-ymin)/2 + ymin > c/2 - th_c_center and (ymax-ymin)/2 + ymin < c/2 + th_c_center):
                dst = cv2.cornerHarris(img_mask,2,3,0.04)
                final = np.zeros((r,c))
                final[dst>0.01*dst.max()]=1
                pictures.append((imagePath[-9:-4], np.sum(final)))
            else:
                pictures.append((imagePath[-9:-4], 0))
        else:
            pictures.append((imagePath[-9:-4], 0))

pictures.sort(key = lambda x: x[1], reverse=True)


histo = [[[ [] for _ in range(bins)] for _ in range(bins)] for _ in range(bins)]


for picture in pictures:
    numb_paint = int(picture[0])
    p = palets[numb_paint]
    
    histo[p[0]][p[1]][p[2]].append(picture[0])
    if len(histo[p[0]][p[1]][p[2]]) >= 5:
        break
print(histo[p[0]][p[1]][p[2]])