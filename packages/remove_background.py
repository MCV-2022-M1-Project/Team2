import numpy as np
from PIL import Image
import cv2
from skimage.filters import threshold_local
from collections import deque
import imutils

class RemoveBackground:
    @staticmethod
    def compute_removal(data):
        # Grab the shape of the image
        row, col, _ = data.shape

        # Define the thresholds
        threshold_rec_col = round(col / 1.5)
        threshold_rec_row = round(row / 1.5)
        edge_r = round(col / 200)
        edge_c = round(row / 200)

        image_gray = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)

        # Blur the image
        blurred = cv2.medianBlur(image_gray, 7)

        # Use dynamic thresholding
        thresh = cv2.Canny(blurred, 10, 100)
        # Every pixel similar to the background color with shadow gets to False
        mask = (thresh > 0).astype("uint8")

        mask[0:edge_r,:] = 0
        mask[-edge_r:,:] = 0
        mask[:,0:edge_c] = 0
        mask[:,-edge_c:] = 0

        # mask = np.logical_not((data[:,:,0] > valor[0] - threshold) & (data[:,:,0] < valor[0]  + threshold) & (data[
        # :,:,1] > valor[1]  - threshold) & (data[:,:,1] < valor[1]  + threshold) & (data[:,:,2] > valor[2]  -
        # threshold) & (data[:,:,2] < valor[2] + threshold))


        # Reconstruct painting by columns
        for j in range(col):
            first = 0
            final = row
            found_first = False

            for i in range(row):
                if mask[i, j] > 0 and not found_first:
                    found_first = True
                    first = i
                if mask[i, j] > 0 and i - final <= threshold_rec_row:
                    final = i
            if found_first:
                mask[first:final, j] = 1
        

        bfs_threh = RemoveBackground.BFS(mask)


        kernel = np.ones((int(row/40), int(col/40)), np.uint8)
        th_open = cv2.morphologyEx(bfs_threh, cv2.MORPH_OPEN, kernel)
        th_open = cv2.morphologyEx(th_open, cv2.MORPH_CLOSE, kernel)

        # Return the mask
        mask = mask.astype("uint8")
        num_labels, labels, stats, centroids =  cv2.connectedComponentsWithStats(th_open)
        stats = sorted(stats[1:], key = lambda t: t[4], reverse=True)

        if len(stats) > 1 and stats[1][4] > row*col/50:
            if len(stats) > 2 and stats[2][4] > row*col/50:
                stats = stats[0:3]
            else:
                stats = stats[0:2]
        else:
            stats = stats[0:1]
        return (th_open, stats)

    @staticmethod
    def compute_removal_2(image):
        # Convert image to gray scale
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Blur the image
        #kernel = np.ones((5,5),np.float32)/(5*5)
        blurred = cv2.medianBlur(image_gray, 7)

        # Use dynamic thresholding
        #T = threshold_local(blurred, 29, offset=5, method="gaussian")
        #thresh = (blurred < T).astype("uint8") * 255

        thresh = cv2.Canny(blurred, 10, 100)
        #thresh = imutils.auto_canny(blurred, 0.3)

        r,c = image_gray.shape

        # Dilate the borders
        kernel = np.ones((5, 5), np.uint8)
        #thresh_dilated = cv2.dilate(thresh, kernel, iterations=1)
        kernel_1 = np.ones((int(r/25), int(8)), np.uint8)
        kernel_2 = np.ones((int(8), int(c/25)), np.uint8)
        kernel = np.ones((int(r/20), int(c/20)), np.uint8)
        th_closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_1)
        th_closed = cv2.morphologyEx(th_closed, cv2.MORPH_CLOSE, kernel_2)

        # Get all the background component
        bfs_threh = RemoveBackground.BFS(th_closed)
 
        # Clean the mask
        kernel = np.ones((int(r/40), int(c/40)), np.uint8)
        th_open = cv2.morphologyEx(bfs_threh, cv2.MORPH_OPEN, kernel)
        th_open = cv2.morphologyEx(th_open, cv2.MORPH_CLOSE, kernel)
 
        # Get 2 biggest bb
        num_labels, labels, stats, centroids =  cv2.connectedComponentsWithStats(th_open)
        stats = sorted(stats[1:], key = lambda t: t[4], reverse=True)
        r,c = image_gray.shape

        if len(stats) > 1 and stats[1][4] > r*c/50:
            if len(stats) > 2 and stats[2][4] > r*c/50:
                stats = stats[0:3]
            else:
                stats = stats[0:2]
        else:
            stats = stats[0:1]
        return (th_open, stats)

    @staticmethod
    def BFS(thresh):
        queue = []
        row, col = thresh.shape
        mask = np.ones((row,col)).astype("uint8")*255
        mask[0,0] = 0
        queue = deque([(0,0)])

        while len(queue) > 0:
            i,j = queue[0]
            queue.popleft()


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
