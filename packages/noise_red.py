import argparse
import collections
import pickle
import cv2
import numpy as np
from imutils.paths import list_images



path = "dataset/qsd1_w3/"

index = {}

class RemoveNoise:
    def __init__(self,image):
        self.image = image
    def checkImage(self):
        #check if image has salt and pepper noise
        m_filtered = cv2.medianBlur(self.image, 3)
        noise = cv2.subtract(m_filtered, self.image)
        #count the noise pixels and check how many pixels there are relative to image size
        n_pix = np.sum(noise > 30)
        if n_pix>((self.image.shape[0]*self.image.shape[1]) / 50):
            new_image = cv2.medianBlur(self.image, 3)
            print("Salt and pepper noise detected!")
            return new_image
        else:
            return self.image
        
        #check if image has other type of noise TODO
