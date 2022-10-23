import argparse
import collections
import pickle
import cv2
import numpy as np
from imutils.paths import list_images




class PSNR:
    def __init__(self, modified):
        self.modified = modified

    def gaussianBlur(self, fr, to):
        res = []
        max_value = -1

        for i in range(fr, to+1, 2):
            image_b = cv2.GaussianBlur(self.modified,(i,i),cv2.BORDER_DEFAULT)
            value = cv2.PSNR(self.modified, image_b)
            if value > max_value:
                max_value = value
                photo = image_b
            res.append(value)
            print("value_gaussianBlur", "tam", i, "_", value)
        print("max_gaussianBlur", max_value)
        return (max_value, photo)
        
    
    def medianBlur(self, fr, to):
        res = []
        max_value = -1

        for i in range(fr, to+1, 2):
            image_b = cv2.medianBlur(self.modified,i)
            value = cv2.PSNR(self.modified, image_b)
            if value > max_value:
                max_value = value
                photo = image_b
            res.append(value)
            print("value_medianBlur", "tam", i, "_", value)
        print("max_medianBlur", max_value)
        return (max_value, photo)
        
            
