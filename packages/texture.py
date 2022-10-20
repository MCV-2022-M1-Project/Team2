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

    def compute_histogram_blocks(self, image, text_box=None, block_size=16):

        if text_box:
            tlx = text_box[0]
            tly = text_box[1]
            brx = text_box[2]
            bry = text_box[3]

        # hist_concatenated = None

        if not text_box:
            histogram = self.compute_hog(image)

        # If there's a text bounding box ignore the pixels inside it
        else:
            img_cell_vector = []

            for x in range(image.shape[1] - 1):
                for y in range(image.shape[0] - 1):
                    if not (tlx < x < brx and tly < y < bry):
                        img_cell_vector.append(image[y, x, :])

            img_cell_vector = np.asarray(img_cell_vector)
            if img_cell_vector.size != 0:
                img_cell_matrix = np.reshape(img_cell_vector, (img_cell_vector.shape[0], 1, -1))
                histogram = self.compute_hog(img_cell_matrix)

        return histogram
