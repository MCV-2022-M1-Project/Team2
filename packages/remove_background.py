import numpy as np
from PIL import Image
import cv2
from skimage.filters import threshold_local

class RemoveBackground:
    @staticmethod
    def compute_removal(data):
        # Define thresholds
        threshold = 65
        threshold_2 = 40

        # Grab the shape of the image
        row, col, _ = data.shape

        # Define the thresholds
        threshold_rec_col = round(col / 1.5)
        threshold_rec_row = round(row / 1.5)
        threshold_erase_col = round(col / 3)
        threshold_erase_row = round(row / 3)

        # Get average color of background taking a sample from each side of the picture
        valor = data[0, round(col / 2), :].astype(int)
        valor = valor + data[row - 1, round(col / 2), :].astype(int)
        valor = valor + data[round(row / 2), 0, :].astype(int)
        valor = valor + data[round(row / 2), col - 1, :].astype(int)
        valor = valor / 4
        col_shadow = valor - 120

        # Every pixel similar to the background color gets to False
        mask = np.logical_not((data[:, :, 0] > valor[0] - threshold) & (data[:, :, 1] > valor[1] - threshold) & (
                    data[:, :, 2] > valor[2] - threshold))

        # Every pixel similar to the background color with shadow gets to False
        mask = np.logical_and(mask, np.logical_not(
            (data[:, :, 0] > col_shadow[0] - threshold_2 + 10) & (data[:, :, 0] < col_shadow[0] + threshold_2) & (
                        data[:, :, 1] > col_shadow[1] - threshold_2 + 10) & (
                        data[:, :, 1] < col_shadow[1] + threshold_2) & (
                        data[:, :, 2] > col_shadow[2] - threshold_2 + 10) & (
                        data[:, :, 2] < col_shadow[2] + threshold_2)))

        # mask = np.logical_not((data[:,:,0] > valor[0] - threshold) & (data[:,:,0] < valor[0]  + threshold) & (data[
        # :,:,1] > valor[1]  - threshold) & (data[:,:,1] < valor[1]  + threshold) & (data[:,:,2] > valor[2]  -
        # threshold) & (data[:,:,2] < valor[2] + threshold))

        # Reconstruct painting by rows
        for i in range(row):
            first = 0
            final = col
            found_first = False

            for j in range(col):
                if mask[i, j] > 0 and not found_first:
                    found_first = True
                    first = j
                if mask[i, j] > 0 and j - final <= threshold_rec_col:
                    final = j
            if found_first:
                mask[i, first:final] = 1
        
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
        
        ######
        # Erase small objects by columns
        for j in range(col):
            first = 0
            final = 0
            conti = False

            for i in range(row):
                if mask[i, j] > 0 and not conti:
                    conti = True
                    first = i
                elif conti and (i == row - 1 or mask[i + 1, j] <= 0):
                    final = i + 1
                    conti = False
                    if final - first <= threshold_erase_row:
                        mask[first:final, j] = 0

        # Erase small objects by rows
        for i in range(row):
            first = 0
            final = 0
            conti = False

            for j in range(col):
                if mask[i, j] > 0 and not conti:
                    conti = True
                    first = j
                elif conti and (j == col - 1 or mask[i, j + 1] <= 0):
                    final = j + 1
                    conti = False
                    if final - first <= threshold_erase_col:
                        mask[i, first:final] = 0

        # Erase again small objects by columns
        for j in range(col):
            first = 0
            final = 0
            conti = False

            for i in range(row):
                if mask[i, j] > 0 and not conti:
                    conti = True
                    first = i
                elif conti and (i == row - 1 or mask[i + 1, j] <= 0):
                    final = i + 1
                    conti = False
                    if final - first <= threshold_erase_row:
                        mask[first:final, j] = 0

        # Return the mask
        mask = mask.astype("uint8")
        num_labels, labels, stats, centroids =  cv2.connectedComponentsWithStats(mask)

        return (mask.astype("uint8"), stats)

    @staticmethod
    def compute_removal_2(image):
        # Convert image to gray scale
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Blur the image
        kernel = np.ones((5,5),np.float32)/(5*5)
        blurred = cv2.filter2D(image_gray,-1,kernel)

        # Use dynamic thresholding
        T = threshold_local(blurred, 29, offset=5, method="gaussian")
        thresh = (blurred < T).astype("uint8") * 255

        # Dilate the borders
        kernel = np.ones((5, 5), np.uint8)
        thresh_dilated = cv2.dilate(thresh, kernel, iterations=1)

        # Get all the background component
        bfs_threh = RemoveBackground.BFS(thresh_dilated)
        """
        num_labels, labels, stats, centroids =  cv2.connectedComponentsWithStatsWithAlgorithm(thresh_dilated+1, connectivity = 8, ltype = cv2.CV_32S, ccltype = cv2.CCL_GRANA)

        # Search the background label
        label = 0
        while stats[label][0] != 0 and stats[label][1] != 0: label += 1

        # Erase the background
        bfs_threh = (np.logical_not(labels == label)*255).astype("uint8")
        """
        
        # Clean the mask
        kernel = np.ones((40, 40), np.uint8)
        th_open = cv2.morphologyEx(bfs_threh, cv2.MORPH_OPEN, kernel)
        th_open = cv2.morphologyEx(th_open, cv2.MORPH_CLOSE, kernel)

        # Get 2 biggest bb
        num_labels, labels, stats, centroids =  cv2.connectedComponentsWithStats(th_open)
        stats = sorted(stats[1:], key = lambda t: t[4], reverse=True)
        r,c = image_gray.shape

        if stats[1][4] > r*c/100:
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
