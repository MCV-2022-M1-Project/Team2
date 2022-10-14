import cv2
import numpy as np
from imutils.paths import list_images
from packages import bb_intersection_over_union
from packages import RemoveText
import pickle

with open('../dataset/qsd1_w2/text_boxes.pkl', 'rb') as f:
    data = pickle.load(f)
i = 0
IOU = []

for (imagePath) in (sorted(list_images("../dataset/qsd1_w2"))):
    if "jpg" in imagePath:
        bb = [data[i][0][0][0], data[i][0][0][1], data[i][0][2][0], data[i][0][2][1]]
        image = cv2.imread(imagePath)
        text_id = RemoveText(image)
        bbox = text_id.extract_text()
        iou = bb_intersection_over_union(bbox, bb)
        IOU.append(iou)
        print("IOU", iou)
        i = i + 1
        image = cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 4)
        #cv2.imshow("image", image)
        #cv2.waitKey(0)

IOU = np.array(IOU)
print("Mean IOU", IOU.mean())
