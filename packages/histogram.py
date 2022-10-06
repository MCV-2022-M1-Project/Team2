# Import required packages
import cv2
import numpy as np


class RGBHistogram:
    def __init__(self, bins):
        # Store the number of bins of the histogram
        self.bins = bins

    def compute_histogram(self, image):
        # Compute a RGB Histogram normalize it
        histogram = cv2.calcHist([image], [0, 1, 2], None, self.bins, [0, 256, 0, 256, 0, 256])

        # Normalize the histogram
        histogram = cv2.normalize(histogram, histogram)

        # Flatten the 3D histogram to 1D
        return histogram.flatten()

    def compute_labHistogram(self, image):
        # Compute a lab Histogram
        image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        histogram = cv2.calcHist([image], [0, 1, 2], None, self.bins, [0, 256, 0, 256, 0, 256])

        # Normalize the histogram
        histogram = cv2.normalize(histogram, histogram)

        # Flatten the 3D histogram to 1D
        return histogram.flatten()

    def compute_hsvHistogram(self, image):
        # Compute an HSV histogram
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        histogram = cv2.calcHist([image], [0, 1, 2], None, self.bins, [0, 180, 0, 256, 0, 256])

        # Normalize the histogram
        histogram = cv2.normalize(histogram, histogram)

        # Flatten the 3D histogram to 1D
        return histogram.flatten()

    def compute_grayscaleHistogram(self, image):
        # Compute a grayscale histogram
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        histogram = cv2.calcHist([image], [0], None, [256], [0, 256])

        # Normalize the histogram
        histogram = cv2.normalize(histogram, histogram)

        # Return the histogram
        return histogram

