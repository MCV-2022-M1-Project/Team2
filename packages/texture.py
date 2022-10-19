# Import required packages
import cv2
import numpy as np
from skimage import feature
from skimage.feature import hog


class TextureDescriptors:
    def compute_hog(self, image):
        # Convert image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Calculate the hog coefficients and multiply by 256
        hog_coefficients = hog(gray, orientations=8, pixels_per_cell=(16, 16),
                               cells_per_block=(1, 1), visualize=False, feature_vector=True)
        hog_coefficients *= 256

        # Convert the coefficients to a histogram
        histogram = cv2.calcHist([hog_coefficients.astype(np.uint8)], [0], None, [8], [0, 256])
        histogram = histogram.astype("float32")

        # Normalize the histogram
        histogram = cv2.normalize(histogram, histogram, alpha=0, beta=1,
                                  norm_type=cv2.NORM_MINMAX)

        # Return the histogram
        return histogram

    def compute_lbp(self, image, numPoints=8, radius=2, eps=1e-7):
        # Convert image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Calculate the Local binary pattern
        lbp = feature.local_binary_pattern(gray, numPoints, radius, method="uniform")

        # Convert them to histogram
        histogram = cv2.calcHist([lbp.astype(np.uint8)], [0], None, [numPoints + 2], [0, numPoints + 2])

        # Normalize the histogram
        histogram = cv2.normalize(histogram, histogram, alpha=0, beta=1,
                                  norm_type=cv2.NORM_MINMAX)

        # Return the histogram
        return histogram
