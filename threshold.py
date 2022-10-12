# Import the necessary packages
import imutils
import numpy as np
from skimage.filters import threshold_local
import argparse
import cv2
import time





def BFS(thresh):
    queue = []
    row, col = thresh.shape
    mask = np.ones((row,col))*255
    mask[0,0] = 0
    queue = [(0,0)]

    while len(queue) > 0:
        i,j = queue[0]
        queue.pop(0)

        """
        if i-1 >= 0 and thresh[i-1, j] <= 0:
            if (i-1,j) not in queue and mask[i-1,j] > 0: 
                queue.append((i-1,j))
                mask[i-1,j] = 0
        """
        if i+1 < row and thresh[i+1, j] <= 0:
            if (i+1,j) not in queue and mask[i+1,j] > 0:
                queue.append((i+1,j))
                mask[i+1,j] = 0
        """
        if j-1 >= 0 and thresh[i, j-1] <= 0:
            if (i,j-1) not in queue and mask[i,j-1] > 0:
                queue.append((i,j-1))
                mask[i,j-1] = 0
        """
        if j+1 < col and thresh[i, j+1] <= 0:
            if (i,j+1) not in queue and mask[i,j+1] > 0:
                queue.append((i,j+1))
                mask[i,j+1] = 0
        
    return mask

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", default="../qsd2_w2/00002.jpg", help="Path to the image")
args = vars(ap.parse_args())

# Load the image, convert it to grayscale, and blur it 
image = cv2.imread(args["image"])
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

kernel = np.ones((5,5),np.float32)/(5*5)
blurred = cv2.filter2D(image,-1,kernel)

#blurred = cv2.GaussianBlur(image, (5, 5), 0)
#cv2.imshow("Image", image)

# Use OpenCv Adaptive threshold
thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 25, 15)
#cv2.imshow("OpenCV Mean Thresh", thresh)

# Use Scikit Learn
T = threshold_local(blurred, 29, offset=5, method="gaussian")
thresh = (blurred < T).astype("uint8") * 255
cv2.imwrite("1_mask_thresh.png", thresh)

kernel = np.ones((10, 10), np.uint8)
threh_dilated = cv2.dilate(thresh, kernel, iterations=1)



start_time = time.time()
bfs_threh = BFS(threh_dilated)
cv2.imwrite("2_mask.png", bfs_threh)
print("--- %s seconds ---" % (time.time() - start_time))

kernel = np.ones((40, 40), np.uint8)
th_open = cv2.morphologyEx(bfs_threh, cv2.MORPH_OPEN, kernel)
th_open = cv2.morphologyEx(th_open, cv2.MORPH_CLOSE, kernel)




cv2.imwrite("3_mask_closed.png", th_open)

