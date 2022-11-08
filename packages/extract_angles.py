import numpy as np
import cv2
from skimage.transform import probabilistic_hough_line


def extract_angle(image):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Blur the image, convert to canny and apply dilation
    gray_blur = cv2.medianBlur(gray, 5)
    edges = cv2.Canny(gray_blur, 50, 200)
    kernel = np.ones((3, 3), np.uint8)
    dilation = cv2.dilate(edges, kernel, iterations=1)

    # Detect lines using hough transform
    lines = probabilistic_hough_line(dilation, threshold=300, line_length=int(np.floor(np.size(image, 0) / 5)), line_gap=3)
    max = 0
    angle = 0
    pos = 0

    # Loop over the lines if not none
    if lines:
        for j in range(0, len(lines)):
            # Grab the current line
            line = lines[j]

            # Correct the line
            x1, y1, x2, y2 = correctCoordinates(line)

            # Calculate the max angle
            val = x1 + y1
            angle_current = getAngle(x1, y1, x2, y2)
            if (val > max or max == 0) and (45 > angle_current > -45):
                angle = angle_current
                max = val
                pos = j
        # x1, y1, x2, y2 = correctCoordinates(lines[pos])
        # cv2.line(img, (x1, y1), (x2, y2), (125), 5)

    # Correct the angles
    if angle > 45:
        angle = 90 - angle
    elif angle < -45:
        angle = angle + 90

    image = rotate(image, angle)

    # Correct the angles
    if angle < 0:
        angle = angle + 180
    elif angle < 90:
        angle = angle + 90

    return angle


def getAngle(x1, y1, x2, y2):
    # Vertical line
    if x2 == x1:
        angle = 90

    # Horizontal line
    elif y1 == y2:
        angle = 0
    else:
        # Calculate the angle
        angle = np.arctan((y2 - y1) / (x2 - x1))
        # Transform to degrees
        angle = angle * 180 / np.pi

    # Anti-clockwise
    return -angle


def rotate(img, angle):
    # Get center coordinates of image
    h = int(round(np.size(img, 0) / 2))
    w = int(round(np.size(img, 1) / 2))
    center = (w, h)

    # Rotate the image
    M = cv2.getRotationMatrix2D(center, -angle, 1)
    result = cv2.warpAffine(img, M, (w * 2, h * 2))

    # Return the result
    return result


def correctCoordinates(line):
    # Grab coordinates of the points on the line
    x2 = line[0][0]
    y2 = line[0][1]
    x1 = line[1][0]
    y1 = line[1][1]

    # If x1 is greater than x2 swap the coordinates
    if x1 > x2:
        temp = x1
        x1 = x2
        x2 = temp
        temp = y1
        y1 = y2
        y2 = temp

    # Return the corrected coordinates
    return x1, y1, x2, y2
