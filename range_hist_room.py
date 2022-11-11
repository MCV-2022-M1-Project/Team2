# Import required packages
import argparse
import cv2
from imutils.paths import list_images
from matplotlib import pyplot as plt
import numpy as np

# Construct argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--index", default="../dataset/bbdd", help="Path to the image dataset")


imagePaths = []
sv = []


for imagePath in sorted(list_images("D:/Documentos/Uni/Master/M1/dataset/BBDD")):
    if "jpg" in imagePath:
        print(imagePath)
        
        imagePaths.append(imagePath)
        image = cv2.imread(imagePath)
        hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)

        o = 0
        r,c = hsv[:,:,1].shape
        o = np.sum(np.logical_and(np.logical_and(hsv[:,:,0] >= 50, hsv[:,:,0] < 70), np.logical_and(hsv[:,:,1] > 80, hsv[:,:,2] > 85)))
        sv.append((float)(o / (r * c)))
        print(sv[-1])

best = []

for i in range(5):
    val = max(sv)
    index = sv.index(val)
    best.append(imagePaths[index])
    imagePaths.pop(index)
    sv.pop(index)

print("Mother nature room:")
for path in best:
    print("\t" + path)
    im = cv2.imread(path)
    plt.imshow(cv2.cvtColor(im,cv2.COLOR_BGR2RGB), 'gray'),plt.show()