import argparse
import collections
import pickle
import cv2
import numpy as np
from imutils.paths import list_images
from packages import RemoveNoise



path = "../dataset/qsd2_w3/"


tp = 0
fn = 0
fp = 0
tn = 0

for imagePath in list_images(path):
    if "jpg" in imagePath and not "non_augmented" in imagePath:
        # Load the image
        print(imagePath)
        image = cv2.imread(imagePath)
        rm_v = RemoveNoise(image)
        tp_a, fn_a, fp_a, tn_a = rm_v.noise_accuracy_qsd2_w3(imagePath[-9:-4])
        tp += tp_a
        fn += fn_a
        fp += fp_a
        tn += tn_a
precission = (tp)/(tp+fp)
recall = (tp)/(tp+fn)
f1 = 2*(precission*recall)/(precission+recall)

print("Precission:", precission)
print("Recall:", recall)
print("F1:", f1)