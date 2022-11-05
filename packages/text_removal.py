# Import required packages
import cv2
import imutils
import numpy as np
from skimage.filters import threshold_local


# Define function to determine the best rectangle
def compute_score(bouding_box, image):
    # Grab dimensions of the image
    image_width = image.shape[1]
    image_width_center = image_width / 2.0
    aspect_ratio = 3.5

    # Grab the coordinates of the bounding box
    x = bouding_box[0]
    w = bouding_box[2]
    h = bouding_box[3]
    cX = x + w / 2.0
    rect_ratio = float(w) / h

    # Calculate x and y center score depending on their value relative to image
    x_center_score = abs(cX - image_width_center)

    # Calculate ratio score
    ratio_score = abs(rect_ratio - aspect_ratio)

    # Give weights to all scores
    x_weight = 0.5
    ratio_weight = 0.5

    # Calculate the final score
    final_Score = x_center_score * x_weight + ratio_score * ratio_weight

    # Return the final score
    return final_Score


# Calculate the best bounding box depending on the score
def get_best_rectangle(rectangles, image):
    # Define list of distances
    distances = []

    # Loop over the rectangles
    for rectangle in rectangles:
        if rectangle is not None:
            distances.append(compute_score(rectangle, image))
        else:
            distances.append(100000)

    # Sort the distances
    idx_best_rectangle = np.argsort(distances)[0]

    # Return the lowest distance
    return rectangles[idx_best_rectangle]


def compute_morphology_roi(morph_image, kernel_roi, sigma=0.025):
    # Crop the image
    left = int(morph_image.shape[1] * sigma)
    right = int(morph_image.shape[1] * (1 - sigma))
    top = int(morph_image.shape[0] * sigma)
    bottom = int(morph_image.shape[0] * (1 - sigma))

    # Compute closing for cropped part
    morph_roi = morph_image[top:bottom, left:right]
    closed_roi = cv2.morphologyEx(morph_roi, cv2.MORPH_CLOSE, kernel_roi)
    closed = morph_image
    closed[top:bottom, left:right] = closed_roi

    # Return the image
    return closed


# Function to apply constraints on the rectangles
def get_constrained_rectangles(image):
    # Grab the contours
    contours = cv2.findContours(image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)

    # Grab the image dimensions and initialize list of rectangles
    (height_image, width_image) = image.shape[:2]
    rectangles = []

    # Loop over the contours
    for contour in contours:
        # Approximate to the rectangle
        x, y, w, h = cv2.boundingRect(contour)

        # Calculate percentage of non zero contour points
        r = float(cv2.countNonZero(image[y:y + h, x:x + w])) / (w * h)

        # Apply constraints on the rectangles
        if (width_image / 10 < w < width_image * 0.95) and (height_image / 40 < h < height_image / 2) and (w > h * 2) \
                and r > 0.35:
            rectangles.append((x, y, w, h))

    # Return the rectangles
    return rectangles


def get_best_box(image, filter1_x, filter1_y, threshold, filter2_x, filter2_y):
    # Initialize the kernel and structuring element
    filterSize = (filter1_x, filter1_y)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, filterSize)

    # Perform tophat and blackhat operations
    tophat = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)
    blackhat = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)

    # Threshold the images
    tophat[(tophat[:, :] < threshold)] = 0
    blackhat[(blackhat[:, :] < threshold)] = 0

    # Initialize the kernel and structuring element
    filterSize = (filter2_x, filter2_y)
    kernel_roi = cv2.getStructuringElement(cv2.MORPH_RECT, filterSize)

    # Compute morphological filtering on the tophat and blackhat
    tophat = compute_morphology_roi(tophat, kernel_roi)
    blackhat = compute_morphology_roi(blackhat, kernel_roi)

    # Get rectangles for tophat and blackhat and add them
    rectangles_tophat = get_constrained_rectangles(tophat)
    rectangles_blackhat = get_constrained_rectangles(blackhat)
    rectangles = rectangles_tophat + rectangles_blackhat

    # If length of rectangles is 0 return none else return only the best rectangle
    if len(rectangles) == 0:
        return None
    else:
        best_rectangle = get_best_rectangle(rectangles, image)

    # Return the rectangle
    return best_rectangle


# Function to expand the rectangles to get the bounding box
def expand_box(img, box, sigma_x=0.005, sigma_y=0.015):
    # Grab the coordinates of the box
    [tlx, tly, brx, bry] = box

    # Calculate hoe much to expand
    tlx_expanded = int(tlx * (1 - sigma_x))
    tly_expanded = int(tly * (1 - sigma_y))
    brx_expanded = int(brx * (1 + sigma_x))
    bry_expanded = int(bry * (1 + sigma_y))

    # Define and return the expanded box
    cv2.rectangle(img, (tlx_expanded, tly_expanded), (brx_expanded, bry_expanded), (255, 0, 0), 3)
    expanded_box = [tlx_expanded, tly_expanded, brx_expanded, bry_expanded]
    return expanded_box


# Detect text
def detect_text_box(image):
    # Convert and split the image to lab and hsv components
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Blur the image
    kernel = np.ones((5, 5), np.float32) / (5 * 5)
    blurred = cv2.filter2D(gray, -1, kernel)

    # Use dynamic thresholding
    T = threshold_local(blurred, 29, offset=5, method="gaussian")
    thresh = (blurred < T).astype("uint8") * 255

    # Define list of best boxes
    best_boxes = [get_best_box(l, 60, 30, 150, 90, 1), get_best_box(a, 60, 30, 25, 90, 1),
                  get_best_box(b, 60, 30, 20, 90, 1),
                  get_best_box(thresh, 30, 30, 150, int(image.shape[1] / 8), int(image.shape[1] / 8)),
                  get_best_box(s, 30, 30, 30, int(image.shape[1] / 8), int(image.shape[1] / 8))]

    # Loop over the best boxes
    for best_box in best_boxes:
        if best_box is not None:
            [x, y, w, h] = best_box
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 5)

    # Calculate the top rectangle from best boxes list
    final_best_box = get_best_rectangle(best_boxes, image)

    # Check whether bounding box is found
    if final_best_box is not None:
        [x, y, w, h] = final_best_box

        final_best_box = [x, y, x + w, y + h]

        [tlx, tly, brx, bry] = final_best_box
        cv2.rectangle(image, (tlx, tly), (brx, bry), (0, 255, 0), 3)

        final_best_box = expand_box(image, final_best_box)

    # Return the bounding box
    return final_best_box


def text_background_detection(image, bbox):
    # Get histogram of bbox area
    hist = [[0] * 256, [0] * 256, [0] * 256]
    for i in range(bbox[0], bbox[2]):
        for j in range(bbox[1], bbox[3]):
            hist[0][image[j][i][0]] += 1
            hist[1][image[j][i][1]] += 1
            hist[2][image[j][i][2]] += 1

    # Get the most represented color in the bbox assuming it will always be the text background
    maxVal = [max(hist[0]), max(hist[1]), max(hist[2])]
    maxIndex = [hist[0].index(maxVal[0]), hist[1].index(maxVal[1]), hist[2].index(maxVal[2])]

    # Get a mask for the color
    err = 1
    textBackground = image * 0
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if maxIndex[0] + err >= image[i][j][0] >= maxIndex[0] - err and maxIndex[1] + err >= \
                    image[i][j][1] >= maxIndex[1] - err and maxIndex[2] + err >= image[i][j][2] >= \
                    maxIndex[2] - err:
                textBackground[i][j] = 255

    # Perform horizontal closing
    kernel_hor = np.ones((1, int(image.shape[1] / 8)), np.uint8)
    textBackground = cv2.dilate(textBackground, kernel_hor, iterations=1)
    textBackground = cv2.erode(textBackground, kernel_hor, iterations=1)

    # Perform vertical closing
    kernel_vert = np.array([[1], [1], [1]])
    textBackground = cv2.dilate(textBackground, kernel_vert, iterations=1)
    textBackground = cv2.erode(textBackground, kernel_vert, iterations=1)

    # Return the processed image
    return (cv2.cvtColor(textBackground, cv2.COLOR_BGR2GRAY) != 0).astype(np.uint8)


def extract_text(image):
    bbox = detect_text_box(image)

    # Filter the bounding boxes
    if bbox is not None:
        textBackground = text_background_detection(bbox, image)
        bbox = detect_text_box(textBackground)

    # Print and return the bounding
    return bbox
