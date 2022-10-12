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
    mask = np.ones((row,col)).astype("uint8")*255
    mask[0,0] = 0
    queue = [(0,0)]

    while len(queue) > 0:
        i,j = queue[0]
        queue.pop(0)


        if i-1 >= 0 and thresh[i-1, j] <= 0:
            if (i-1,j) not in queue and mask[i-1,j] > 0: 
                queue.append((i-1,j))
                mask[i-1,j] = 0

        if i+1 < row and thresh[i+1, j] <= 0:
            if (i+1,j) not in queue and mask[i+1,j] > 0:
                queue.append((i+1,j))
                mask[i+1,j] = 0

        if j-1 >= 0 and thresh[i, j-1] <= 0:
            if (i,j-1) not in queue and mask[i,j-1] > 0:
                queue.append((i,j-1))
                mask[i,j-1] = 0

        if j+1 < col and thresh[i, j+1] <= 0:
            if (i,j+1) not in queue and mask[i,j+1] > 0:
                queue.append((i,j+1))
                mask[i,j+1] = 0
        
    return mask

def count_painting(mask):
    row, col = thresh.shape
    row_see = row//2
    col_see = col//2
    count_v = 0
    count_h = 0
    cont = False
    for j in range(col):
        if not cont and mask[row_see, j] > 0:
            cont = True
            count_h += 1
        elif cont and mask[row_see, j] <= 0:
            cont = False


# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", default="../qsd2_w2/00009.jpg", help="Path to the image")
args = vars(ap.parse_args())

# Load the image, convert it to grayscale, and blur it 
image = cv2.imread(args["image"])
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

kernel = np.ones((5,5),np.float32)/(5*5)
blurred = cv2.filter2D(image_gray,-1,kernel)

#blurred = cv2.GaussianBlur(image, (5, 5), 0)
#cv2.imshow("Image", image)

# Use OpenCv Adaptive threshold
#thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 25, 15)
#cv2.imshow("OpenCV Mean Thresh", thresh)

# Use Scikit Learn
T = threshold_local(blurred, 29, offset=5, method="gaussian")
thresh = (blurred < T).astype("uint8") * 255
cv2.imwrite("1_mask_thresh.png", thresh)

kernel = np.ones((5, 5), np.uint8)
thresh_dilated = cv2.dilate(thresh, kernel, iterations=1)



start_time = time.time()
bfs_threh = BFS(thresh_dilated)
cv2.imwrite("2_mask.png", bfs_threh)
print("--- %s seconds ---" % (time.time() - start_time))

kernel = np.ones((40, 40), np.uint8)
th_open = cv2.morphologyEx(bfs_threh, cv2.MORPH_OPEN, kernel)
th_open = cv2.morphologyEx(th_open, cv2.MORPH_CLOSE, kernel)




cv2.imwrite("3_mask_closed.png", th_open)

num_labels, labels, stats, centroids =  cv2.connectedComponentsWithStats(th_open)

for i in range(1,len(stats)):
    bb = stats[i]
    
    cv2.imwrite(str(i)+"_mask_res.png", th_open[bb[1]:bb[1]+bb[4],bb[0]:bb[0]+bb[2]])
    cv2.imwrite(str(i)+"_paint_res.png", image[bb[1]:bb[1]+bb[4],bb[0]:bb[0]+bb[2],:])
