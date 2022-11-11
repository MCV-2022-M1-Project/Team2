# Import required packages
import argparse
import cv2
from imutils.paths import list_images
from matplotlib import pyplot as plt

# Construct argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--index", default="../dataset/bbdd", help="Path to the image dataset")



pictures = []
palets = []
bins = 3
div = 256/bins
imagePaths = []
sv = []


for imagePath in sorted(list_images("D:/Documentos/Uni/Master/M1/dataset/BBDD")):
    if "jpg" in imagePath:
        print(imagePath)
        
        imagePaths.append(imagePath)
        image = cv2.imread(imagePath)
        hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)

        v = 0
        s = 0
        count = 0
        for i in range(hsv.shape[0]):
            for j in range(hsv.shape[1]):
                v += hsv[i][j][2]
                s += hsv[i][j][1]
                count += 1
        sv.append((int)(v / count) + (int)(s / count))

svsort = sv
svsort.sort()
best = []

for val in svsort[:5]:
    index = sv.index(val)
    best.append(imagePaths[index])
    imagePaths.pop(index)
    sv.pop(index)

print("Best:")
for path in best:
    print("\t" + path)
    plt.imshow(cv2.imread(path), 'gray'),plt.show()