# Import the necessary packages
import argparse
import pickle

import cv2

from packages import RemoveText, RemoveBackground, HistogramDescriptor, Searcher

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", default="../dataset/qsd1_w2/00002.jpg", help="Path to the image")
args = vars(ap.parse_args())

# Load the image, convert it to grayscale, and blur it 
image = cv2.imread(args["image"])

th_open, stats = RemoveBackground.compute_removal_2(image)
desc = HistogramDescriptor([8, 8, 8])
index = pickle.loads(open("index.pkl", "rb").read())

for i in range(0, len(stats)):
    bb = stats[i]
    image1 = image.copy()

    # cv2.imwrite(str(i) + "_mask_res.png", th_open[bb[1]:bb[1] + bb[4], bb[0]:bb[0] + bb[2]])
    # cv2.imwrite(str(i) + "_paint_res.png", image[bb[1]:bb[1] + bb[4], bb[0]:bb[0] + bb[2], :])

    text_id = RemoveText(image[bb[1]:bb[1] + bb[4], bb[0]:bb[0] + bb[2], :])
    bbox = text_id.extract_text()

    image_rec = cv2.rectangle(image[bb[1]:bb[1] + bb[4], bb[0]:bb[0] + bb[2], :], (bbox[0], bbox[1]),
                              (bbox[2], bbox[3]), (0, 0, 255), 8)

    cv2.imshow("image", image[bb[1]:bb[1] + bb[4], bb[0]:bb[0] + bb[2], :])
    mean, std = cv2.meanStdDev(image1)
    queryFeatures = desc.computeHSV(image1)

    # Perform the search
    searcher = Searcher(index)
    results = searcher.search(queryFeatures)
    predicted_query = []

    # Loop over the top ten results
    for j in range(0, 10):
        # Grab the result
        (score, imageName) = results[j]
        predicted_query.append(int(imageName.replace(".jpg", "")))
        print("\t{}. {} : {:.3f}".format(j + 1, imageName, score))

