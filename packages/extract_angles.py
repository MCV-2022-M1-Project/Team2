import numpy as np
import cv2
from skimage.transform import probabilistic_hough_line
import vg

def extract_angle(image):
    # Convert image to grayscale
    #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Blur the image, convert to canny and apply dilation
    #gray_blur = cv2.medianBlur(gray, 5)
    edges = cv2.Canny(image, 50, 200)
    kernel = np.ones((3, 3), np.uint8)
    dilation = cv2.dilate(edges, kernel, iterations=1)
    row, col = dilation.shape

    # Detect lines using hough transform
    lines = cv2.HoughLinesP(dilation, rho=1., theta=np.pi/180.,
                        threshold=80, minLineLength=30, maxLineGap=10.)
    best_line = [0,0,0,0]
    angle_current = 0
    count = 0
    # Loop over the lines if not none
    if len(lines) > 0:
        best_line = lines[0]
        
        for j in range(0, len(lines)):
            line = lines[j] 
            # Correct the line
            x1, y1, x2, y2 = correctCoordinates(line)

            #image = cv2.line(image,(x1,y1),(x2,y2),(200,200,200),9)
            if abs(y2-y1) <= abs(x2-x1):
                #image_p = cv2.line(image,(x1,y1),(x2,y2),(200,200,200),9)
                angle_current += angle_between([x1+5 - x1, y1 - y1, 0], [x2 - x1, y2 - y1, 0])
                count += 1

    angle = angle_current/count
    x1, y1, x2, y2 = correctCoordinates(best_line)
    #image_p = cv2.line(image,(x1,y1),(x2,y2),(200,200,200),9)
    if angle < 0:
        angle = 180 - angle 
    #cv2.imwrite("image_p.png",image_p)
    #cv2.imwrite("mask_line.png",image)
    image = rotate(image, angle)

    # Correct the angles
    if angle < 0:
        angle = 180 + angle

    return angle
def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    v1 = np.array(v1)
    v2 = np.array(v2)
    angle = vg.signed_angle(v2, v1, look=vg.basis.z)
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
        #angle = abs(angle) % 180
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
    x1 = line[0][2]
    y1 = line[0][3]

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
