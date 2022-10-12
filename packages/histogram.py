# Import required packages
import cv2
import numpy as np


class RGBHistogram:
    def __init__(self, bins, mask):
        # Store the number of bins of the histogram
        self.bins = bins
        self.mask = mask

    def compute_histogram(self, image):
        # Compute a RGB Histogram normalize it
        histogram = cv2.calcHist([image], [0, 1, 2], self.mask, self.bins, [0, 256, 0, 256, 0, 256])

        # Normalize the histogram
        histogram = cv2.normalize(histogram, histogram)

        # Flatten the 3D histogram to 1D
        return histogram.flatten()

    def compute_labHistogram(self, image):
        # Compute a lab Histogram
        image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        histogram = cv2.calcHist([image], [0, 1, 2], self.mask, self.bins, [0, 256, 0, 256, 0, 256])

        # Normalize the histogram
        histogram = cv2.normalize(histogram, histogram)

        # Flatten the 3D histogram to 1D
        return histogram.flatten()

    def compute_hsvHistogram(self, image):
        # Compute an HSV histogram
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        histogram = cv2.calcHist([image], [0, 1, 2], self.mask, self.bins, [0, 180, 0, 256, 0, 256])

        # Normalize the histogram
        histogram = cv2.normalize(histogram, histogram)

        # Flatten the 3D histogram to 1D
        return histogram.flatten()

    def compute_grayscaleHistogram(self, image):
        # Compute a grayscale histogram
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        histogram = cv2.calcHist([image], [0], self.mask, [256], [0, 256])

        # Normalize the histogram
        histogram = cv2.normalize(histogram, histogram)

        # Return the histogram
        return histogram


class HistogramDescriptor:
    def __init__(self, bins):
        # store the number of bins for the histogram
        self.bins = bins

    def computeHSV(self, image):
        # Convert the image to HSV color space and initialize list for storing features
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        features = []

        # Grab dimensions of the image and compute coordinates of the center
        (h, w) = image.shape[:2]
        (cX, cY) = (int(w*0.5), int(h*0.5))

        # Divide the image into 4 regions
        segments = [(0, cX, 0, cY), (cX, w, 0, cY), (cX, w, cY, h), (0, cX, cY, h)]

        # Construct elliptical mask in the center of the image
        (axesX, axesY) = (int(w*0.75)//2, int(h*0.75)//2)
        ellipseMask = np.zeros(image.shape[:2], dtype="uint8")
        cv2.ellipse(ellipseMask, (cX, cY), (axesX, axesY), 0, 0, 360, 255, -1)

        # Loop over the segments
        for (startX, endX, startY, endY) in segments:
            # Construct a mask for each corner of the image, subtracting
            # the elliptical center from it
            cornerMask = np.zeros(image.shape[:2], dtype="uint8")
            cv2.rectangle(cornerMask, (startX, startY), (endX, endY), 255, -1)
            cornerMask = cv2.subtract(cornerMask, ellipseMask)

            # Extract a color histogram and update features list
            histogram = self.histogram(image, cornerMask)
            features.extend(histogram)

        # Extract a color histogram from the elliptical region and update the feature vector
        histogram = self.histogram(image, ellipseMask)
        features.extend(histogram)

        # Return the feature vector
        return np.array(features)

    def histogram(self, image, mask=None):
        # Compute a color Histogram normalize it
        histogram = cv2.calcHist([image], [0, 1, 2], mask, self.bins, [0, 256, 0, 256, 0, 256])

        # Normalize the histogram
        histogram = cv2.normalize(histogram, histogram)

        # Flatten the 3D histogram to 1D
        return histogram.flatten()
