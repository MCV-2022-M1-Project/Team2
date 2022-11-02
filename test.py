import cv2
import numpy as np
from imutils.paths import list_images
from packages import bb_intersection_over_union
from packages import RemoveText, RemoveBackground, RemoveNoise, detect_text_box
import pickle
import math
import os
from skimage.restoration import (denoise_wavelet, estimate_sigma)
import time

def oneImagePaintingW2(data, IOU):
    i = 0
    for (imagePath) in (sorted(list_images("../dataset/qsd1_w2"))):
        if "jpg" in imagePath:
            bb = [data[i][0][0][0], data[i][0][0][1], data[i][0][2][0], data[i][0][2][1]]
            image = cv2.imread(imagePath)
            """text_id = RemoveText(image)
            bbox = text_id.extract_text()"""
            bbox = detect_text_box(image)
            if bbox is None:
                bbox = [0, 0, 0, 0]
            iou = bb_intersection_over_union(bbox, bb)
            IOU.append(iou)
            print("IOU", iou)
            i = i + 1
            image = cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 4)
            # cv2.imshow("image", image)
            #  cv2.waitKey(0)


def twoImagePaintingW2(data, IOU):
    i = 0
    bbs = []
    for (imagePath) in (sorted(list_images("../dataset/qsd1_w4"))):
        if "jpg" in imagePath:
            print("#####")
            image = cv2.imread(imagePath)

            th_open, stats = RemoveBackground.compute_removal_2(image)
            # cv2.imwrite(folder+imagePath[-9:-3]+".png", th_open)
            bb_gts = []

            for k in range(len(data[i])):
                bb_gt_n = [data[i][k][0], data[i][k][1], data[i][k][2], data[i][k][3]]
                bb_gts.append(bb_gt_n)

            bb_aux = []

            for k in range(0, len(stats)):

                bb = stats[k]

                """text_id = RemoveText(image[bb[1]:bb[1] + bb[4], bb[0]:bb[0] + bb[2], :])
                bbox = text_id.extract_text()"""
                rn = RemoveNoise(image[bb[1]:bb[1] + bb[4], bb[0]:bb[0] + bb[2], :])
                queryImage_rn = rn.denoise_image()
                bbox = detect_text_box(queryImage_rn)
                if bbox is None:
                    bbox = [0, 0, 0, 0]

                bb_ph_absolute = [bbox[0] + bb[0], bbox[1] + bb[1], bbox[2] + bb[0], bbox[3] + bb[1]]
                bb_aux.append(bb_ph_absolute)

                if len(bb_gts) > 1:
                    bb_gt = bb_near(bb_gts[0], bb_gts[1], bb_ph_absolute)
                else:
                    bb_gt = bb_gts[0]
                print("bb_ph_absolute", bb_ph_absolute)
                print("bb_gt", bb_gt)

                iou = bb_intersection_over_union(bb_ph_absolute, bb_gt)
                IOU.append(iou)
                print("IOU", iou)
                print("&&&&")

                image = cv2.rectangle(image, (bb_ph_absolute[0], bb_ph_absolute[1]),
                                      (bb_ph_absolute[2], bb_ph_absolute[3]), (0, 0, 255), 4)

            # cv2.imwrite('test/'+imagePath[-9:], image)
            bbs.append(bb_aux)

            i = i + 1


def oneImagePaintingW3(data, IOU):
    i = 0
    for (imagePath) in (sorted(list_images("../dataset/qsd1_w3"))):
        if "jpg" in imagePath and "non_augmented" in imagePath:
            bb = []
            bb.append(data[i][0][0])
            bb.append(data[i][0][1])
            bb.append(data[i][0][2])
            bb.append(data[i][0][3])
            image = cv2.imread(imagePath)
            rm_n = RemoveNoise(image)
            image = rm_n.denoise_image()
            text_id = RemoveText(image)
            bbox = text_id.extract_text()
            iou = bb_intersection_over_union(bbox, bb)
            IOU.append(iou)
            print("IOU", iou)
            i = i + 1
            image = cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 4)
            cv2.imwrite("./bb/" + imagePath[-9:], image)


def bb_near(bb_1, bb_2, bb):
    dist_1 = math.hypot(bb_1[0] - bb[0], bb_1[1] - bb[1])
    dist_2 = math.hypot(bb_2[0] - bb[0], bb_2[1] - bb[1])
    if dist_1 <= dist_2:
        return bb_1
    else:
        return bb_2



def prf():
    # Initialize the parameters and counters
    sumPrecision1 = 0
    sumRecall1 = 0
    sumF11 = 0
    counter1 = 0

    for imagePath in (sorted(list_images("../dataset/qsd1_w4"))):

        if "jpg" in imagePath:
            image = cv2.imread(imagePath)
            mask, stats = RemoveBackground.compute_removal_2(image)
            
            # Save the path to the mask and get directions to original mask
            ogMask = cv2.imread(imagePath[:-3] + "png")
            print(imagePath[:-3] + "png")
            height, width, _ = ogMask.shape

            # Initialize the precision parameters
            tp1 = 0
            fp1 = 0
            fn1 = 0

            # Loop over the original mask
            for i in range(height):
                for j in range(width):
                    if ogMask[i, j, 0] == 0 and mask[i, j] != 0:
                        fp1 += 1
                    elif ogMask[i, j, 0] != 0 and mask[i, j] == 0:
                        fn1 += 1
                    elif ogMask[i, j, 0] != 0 and mask[i, j] != 0:
                        tp1 += 1

            # Calculate the parameters
            precision1 = tp1 / (tp1 + fp1)
            recall1 = tp1 / (tp1 + fn1)
            f11 = 2 * precision1 * recall1 / (precision1 + recall1)

            # Add the parameters
            sumPrecision1 += precision1
            sumRecall1 += recall1
            sumF11 += f11
            counter1 += 1

            # Take the average
            avgPrecision1 = sumPrecision1 / (counter1)
            avgRecall1 = sumRecall1 / (counter1)
            avgF11 = sumF11 / (counter1)

            # Print the values
            print("Method 1 Precision: ", precision1 * 100, "%")
            print("Method 1 Recall: ", recall1 * 100, "%")
            print("Method 1 F1: ", f11 * 100, "%")
            print("Method 1 avg Precision: ", avgPrecision1 * 100, "%")
            print("Method 1 avg Recall: ", avgRecall1 * 100, "%")
            print("Method 1 avg F1: ", avgF11 * 100, "%")
            print()

def main():
    start_time = time.time()
    with open('../dataset/qsd2_w2/text_boxes.pkl', 'rb') as f:
        data = pickle.load(f)
    """
    IOU = []
    oneImagePaintingW3(data, IOU)
    #twoImagePaintingW2()

    IOU = np.array(IOU)
    print("Mean IOU", IOU.mean())
    """
    prf()
    print("--- %s seconds ---" % (time.time() - start_time))

if __name__ == "__main__":
    main()