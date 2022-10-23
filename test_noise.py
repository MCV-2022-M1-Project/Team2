import argparse
import collections
import pickle
import cv2
import numpy as np
from imutils.paths import list_images
from packages import RemoveNoise



path = "../dataset/qsd1_w3/"

count = 0
total = 0
for imagePath in list_images(path):
    if "jpg" in imagePath and not "non_augmented" in imagePath:
        # Load the image
        print(imagePath)
        image = cv2.imread(imagePath)
        rm_v = RemoveNoise(image)
        if rm_v.noise_accuracy_qsd1_w3(imagePath[-9:-4]):
            count += 1
        total += 1 
print("Accuracy:", count*100/total)