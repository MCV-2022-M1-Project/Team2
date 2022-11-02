# Import required packages
import numpy as np


class DetectAndDescribe:
    def __init__(self, detector, descriptor):
        # Store the keypoint detector and local invariant descriptor
        self.detector = detector
        self.descriptor = descriptor

    def describe(self, image, useKpList=True):
        # Detect keyPoints in the image and extract local invariant descriptors
        keyPoints = self.detector.detect(image)
        (keyPoints, descriptors) = self.descriptor.compute(image, keyPoints)

        # If there are no keyPoints or descriptors, return None
        if len(keyPoints) == 0:
            return None, None

        # Check to see if the keyPoints should be converted to a NumPy array
        if useKpList:
            keyPoints = np.int0([kp.pt for kp in keyPoints])

        # Return a tuple of the keyPoints and descriptors
        return keyPoints, descriptors
