import argparse
import collections
import pickle
import cv2
import numpy as np
from imutils.paths import list_images
from packages import RemoveNoise



path = "dataset/qsd1_w3/"

   
for imagePath in list_images(path):
    if "jpg" in imagePath:
        # Extract our unique image ID (i.e. the filename)
        path = imagePath[imagePath.rfind("_") + 1:]

        # Load the image
        image = cv2.imread(imagePath)
        #check the image for noise
        check = RemoveNoise(image)
        
        if check.checkImage:
            cv2.imshow("s",check.checkImage())

        cv2.waitKey(300000)
            