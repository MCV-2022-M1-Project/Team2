# Import required packages
import cv2


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

