import argparse
import collections
import pickle
import cv2
import numpy as np
from imutils.paths import list_images
from packages import RemoveNoise



path = "dataset/qsd1_w3/"

#use this to see a side by side comparison of the original image and the added noise   
for imagePath in list_images(path):
    if "jpg" in imagePath:
        # Extract our unique image ID (i.e. the filename)
        path = imagePath[imagePath.rfind("_") + 1:]

        # Load the image
        image = cv2.imread(imagePath)
        #check gaussian TODO
            
        #check salt and pepper
        m_filtered = cv2.medianBlur(image, 3)
        noise = cv2.subtract(m_filtered, image)
        #count the noise pixels and check how many pixels there are relative to image size
        n_pix = np.sum(noise > 30)
        if n_pix>((image.shape[0]*image.shape[1]) / 50): #values may be changes for a stricter check
            #new_image = cv2.medianBlur(image, 3)
            print("Salt and pepper noise detected!")
            numpy_horizontal_concat = np.concatenate((image, noise), axis=1)
            cv2.destroyAllWindows()
            cv2.imshow('Numpy Horizontal Concat', numpy_horizontal_concat)
            
        else:
            cv2.destroyAllWindows()
            cv2.imshow("Original", image)
        cv2.waitKey(300000)
            