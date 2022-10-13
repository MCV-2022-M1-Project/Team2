import cv2
from imutils.paths import list_images

from packages import RemoveText

for imagePath in list_images("../dataset/qsd1_w2"):
    if "jpg" in imagePath:
        image = cv2.imread(imagePath)
        text_id = RemoveText(image)
        bbox = text_id.text_extraction()

        image = cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 4)
        cv2.imshow("image", image)
        cv2.waitKey(0)
