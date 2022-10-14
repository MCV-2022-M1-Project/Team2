# Import the necessary packages
import imutils
import numpy as np
from skimage.filters import threshold_local
import argparse
import cv2
import time

from packages import RemoveText, RemoveBackground

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", default="./qsd2_w2/00009.jpg", help="Path to the image")
args = vars(ap.parse_args())

# Load the image, convert it to grayscale, and blur it 
image = cv2.imread(args["image"])

th_open, stats = RemoveBackground.compute_removal(image)

for i in range(1,len(stats)):
    bb = stats[i]
    
    cv2.imwrite(str(i)+"_mask_res.png", th_open[bb[1]:bb[1]+bb[4],bb[0]:bb[0]+bb[2]])
    cv2.imwrite(str(i)+"_paint_res.png", image[bb[1]:bb[1]+bb[4],bb[0]:bb[0]+bb[2],:])
    
    
    text_id = RemoveText(image[bb[1]:bb[1]+bb[4],bb[0]:bb[0]+bb[2],:])
    bbox = text_id.text_extraction()
    
    image_rec = cv2.rectangle(image[bb[1]:bb[1]+bb[4],bb[0]:bb[0]+bb[2],:], (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 4)
    cv2.imwrite('test/'+str(i)+"_"+args["image"][-9:], image_rec)
