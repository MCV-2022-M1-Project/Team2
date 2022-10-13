# Import required packages
import numpy as np
import cv2
from skimage.filters import unsharp_mask


class RemoveText:
    def __init__(self, image):
        self.image = image

    def preProcess(self):
        # Define Kernel size and perform topHat and blackhat operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 30))
        tophat = cv2.morphologyEx(self.image, cv2.MORPH_TOPHAT, kernel)
        blackhat = cv2.morphologyEx(self.image, cv2.MORPH_BLACKHAT, kernel)

        # Define a threshold and make the values below it 0
        threshold = 150
        tophat[(tophat[:, :, 0] < threshold) | (tophat[:, :, 1] < threshold) | (tophat[:, :, 2] < threshold)] = (
        0, 0, 0)
        blackhat[
            (blackhat[:, :, 0] < threshold) | (blackhat[:, :, 1] < threshold) | (blackhat[:, :, 2] < threshold)] = (
        0, 0, 0)

        # Perform a series of erosions and dilations on the tophat and blackhat
        kernelzp = np.ones((1, int(self.image.shape[1] / 8)), np.uint8)
        # kernelzp = cv2.getStructuringElement(cv2.MORPH_RECT, (int(self.image.shape[1] / 8),
        # int(self.image.shape[1] / 8)))

        tophat = cv2.dilate(tophat, kernelzp, iterations=1)
        tophat = cv2.erode(tophat, kernelzp, iterations=1)

        # blackhat = cv2.dilate(tophat, kernelzp, iterations=1)
        # blackhat = cv2.erode(tophat, kernelzp, iterations=1)

        # Sum the operations
        img_sum = tophat + blackhat

        # Apply an unsharp mask to sum of tophat and blackhat
        im_sharped = img_sum
        for i in range(3):
            im_sharped[..., i] = unsharp_mask(img_sum[..., i], radius=40, amount=1.2)

        # Return the processed image
        return (cv2.cvtColor(im_sharped, cv2.COLOR_BGR2GRAY) != 0).astype(np.uint8)

    def textSearch(self, thresh):
        # Apply connected component analysis on the image
        output = cv2.connectedComponentsWithStats(thresh, 4, cv2.CV_32S)

        # Store the values of connected component analysis
        (numLabels, labels, stats, centroids) = output

        # Store the areas of the components
        areas = []

        # Loop from 1 since 0 is background
        for f in range(1, numLabels):
            areas.append(stats[f][4])

        # Grab the dimensions of the image and calculate the area
        w, h = self.image.shape[:2]
        areaImage = w * h

        # Define value and index for maximum area
        maxArea_value = 0
        maxArea_index = 0

        # Loop over the number of connected components
        for h in range(0, numLabels - 1):
            # First one is background so skip it
            area1 = stats[h + 1][2] * stats[h + 1][3]
            if area1 > (areaImage / 2):
                continue

            if maxArea_value < areas[h]:
                maxArea_value = areas[h]

        if areas:
            maxArea_index = areas.index(maxArea_value) + 1

        x1 = stats[maxArea_index][0]
        y1 = stats[maxArea_index][1]
        w1 = stats[maxArea_index][2]
        h1 = stats[maxArea_index][3]
        bbox = [x1, y1, x1 + w1, y1 + h1]

        return bbox

    def text_extraction(self):
        sum = self.preProcess()
        (T, threshInv) = cv2.threshold(sum, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        bbox = self.textSearch(threshInv)

        print('[INFO] BBOX: ', bbox)
        print(" ")
        return bbox
