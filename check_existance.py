import numpy as np
import cv2 as cv2
from matplotlib import pyplot as plt
import argparse
import os
from imutils.paths import list_images
from packages.remove_background import RemoveBackground


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--index", default="D:/Documentos/Uni/Master/M1/dataset/BBDD", help="Path to the image dataset")
ap.add_argument("-q", "--query", default="D:/Documentos/Uni/Master/M1/dataset/qsd1_w4", help="Path to the query image")
args = vars(ap.parse_args())

def sift_flann(img1, img2):
    # Initiate SIFT detector 
    sift = cv2.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)

    # FLANN matcher
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1,des2,k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)
    
    return good, kp1, kp2


MIN_MATCH_PERCENTAGE = 0.1

# Initiate SIFT detector 
sift = cv2.SIFT_create()

# Use list_images to grab the image paths and loop over them
for imagePath in list_images(args["query"]):
    if "jpg" in imagePath:
        print(imagePath)
        img_q = cv2.imread(imagePath)

        th_open, stats = RemoveBackground.compute_removal(img_q)
        #plt.imshow(th_open,),plt.show()
        #plt.imshow(img_q,),plt.show()

        for i in range(len(stats)):
            bb_img = img_q[stats[i][0]:stats[i][2], stats[i][1]:stats[i][3]]
            # find the keypoints and descriptors with SIFT
            kp1, des1 = sift.detectAndCompute(bb_img,None)


            count = 0

            for bbddPath in list_images(args["index"]):
                if "jpg" in bbddPath:
                    img_d = cv2.imread(bbddPath)
                    
                    # find the keypoints and descriptors with SIFT
                    kp2, des2 = sift.detectAndCompute(img_d,None)

                    # FLANN matcher
                    FLANN_INDEX_KDTREE = 1
                    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
                    search_params = dict(checks = 50)

                    flann = cv2.FlannBasedMatcher(index_params, search_params)

                    matches = flann.knnMatch(des1,des2,k=2)

                    # store all the good matches as per Lowe's ratio test.
                    good = []
                    for m,n in matches:
                        if m.distance < 0.7*n.distance:
                            good.append(m)
                    
                    #good, kp1, kp2 = sift_flann(img_d, bb_img)
                    #print("{} has {} keypoints".format(imagePath, len(kp2)))

                    min_match_count = int(len(kp2) * MIN_MATCH_PERCENTAGE)
                    
                    if len(good) > min_match_count:
                        print( "{}- {} matches found".format(count, len(good)) )
                        draw_params = dict(matchColor = (0,255,0), singlePointColor = None, matchesMask = None, flags = 2)
                    
                        img3 = cv2.drawMatches(img_d,kp1,bb_img,kp2,good,None,**draw_params)
                        plt.imshow(img3,),plt.show()
                        
                    else:
                        print( "{}- Not enough matches are found - {}/{}".format(count, len(good), min_match_count) )
                        matchesMask = None
                    
                    count += 1
                
