import cv2
import numpy as np
from imutils.paths import list_images
from packages import bb_intersection_over_union
from packages import RemoveText, RemoveBackground
import pickle
import math

def oneImagePainting(data, IOU):
    i = 0
    for (imagePath) in (sorted(list_images("./qsd1_w2"))):
        if "jpg" in imagePath:
            bb = []
            bb.append(data[i][0][0][0])
            bb.append(data[i][0][0][1])
            bb.append(data[i][0][2][0])
            bb.append(data[i][0][2][1])
            image = cv2.imread(imagePath)
            text_id = RemoveText(image)
            bbox = text_id.extract_text()
            iou = bb_intersection_over_union(bbox, bb)
            IOU.append(iou)
            print("IOU", iou)
            i = i + 1
            image = cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 4)
            # cv2.imshow("image", image)
            #  cv2.waitKey(0)

def twoImagePainting(data, IOU):
    i = 0
    for (imagePath) in (sorted(list_images("./qsd2_w2"))):
        if "jpg" in imagePath:
            print("#####")
            image = cv2.imread(imagePath)

            th_open, stats = RemoveBackground.compute_removal_2(image)
            bb_gts = []
            for k in range(len(data[i])):
                bb_gt_n = []
                bb_gt_n.append(data[i][k][0])
                bb_gt_n.append(data[i][k][1])
                bb_gt_n.append(data[i][k][2])
                bb_gt_n.append(data[i][k][3])
                bb_gts.append(bb_gt_n)

            for k in range(0,len(stats)):

                bb = stats[k]
                
                text_id = RemoveText(image[bb[1]:bb[1]+bb[4],bb[0]:bb[0]+bb[2],:])
                bbox = text_id.text_extraction()
                bb_ph_absolute = [bbox[0]+bb[0], bbox[1]+bb[1], bbox[2]+bb[0], bbox[3]+bb[1]]

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

                image = cv2.rectangle(image, (bb_ph_absolute[0], bb_ph_absolute[1]), (bb_ph_absolute[2], bb_ph_absolute[3]), (0, 0, 255), 4)
            cv2.imwrite('test/'+imagePath[-9:], image)

            i = i + 1

def bb_near(bb_1, bb_2, bb):
    dist_1 = math.hypot(bb_1[0] - bb[0], bb_1[1] - bb[1])
    dist_2 = math.hypot(bb_2[0] - bb[0], bb_2[1] - bb[1])
    if dist_1 <= dist_2:
        return bb_1
    else:
        return bb_2
    
def main():
    with open('./qsd2_w2/text_boxes.pkl', 'rb') as f:
        data = pickle.load(f)

    IOU = []
    #oneImagePainting(data, IOU)
    twoImagePainting(data, IOU)
    IOU = np.array(IOU)
    print("Mean IOU", IOU.mean())

if __name__ == "__main__":
    main()
