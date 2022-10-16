# Evaluate the predictions
import argparse
import collections
import pickle
import cv2
from imutils.paths import list_images

from packages import HistogramDescriptor, RemoveText, Searcher, RGBHistogram, RemoveBackground
from packages.average_precicion import mapk


def evaluate(predicted, ground_truth, k):
    file = open(ground_truth, 'rb')
    actual = pickle.load(file)
    result = mapk(actual=actual, predicted=predicted, k=k)
    return result


# Construct argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--index", default="../dataset/qsd2_w2", help="Path to the image dataset")
ap.add_argument("-q1", "--query1", default="../dataset/qsd1_w2", help="Path to the query image")
ap.add_argument("-q2", "--query2", default="../dataset/qsd2_w2", help="Path to the query image")

args = vars(ap.parse_args())

# Initialize a Dictionary to store our images and features
index = {}

# Initialize image descriptor
descriptor = HistogramDescriptor((8, 8, 8))
print("Indexing images")

# Use list_images to grab the image paths and loop over them
for imagePath in list_images(args["index"]):
    if "jpg" in imagePath:
        # Extract our unique image ID (i.e. the filename)
        path = imagePath[imagePath.rfind("_") + 1:]

        # Load the image, compute histogram and update the index
        image = cv2.imread(imagePath)
        features = descriptor.computeHSV(image)
        index[path] = features

# Sort the dictionary according to the keys
index = collections.OrderedDict(sorted(index.items()))
predicted = []
Results = []
bounding_boxes = []

# Load the query images
for imagePath in sorted(list_images(args["query1"])):
    if "jpg" in imagePath:
        queryImage = cv2.imread(imagePath)
        print("query: {}".format(imagePath))

        text_id = RemoveText(queryImage)
        bbox = text_id.extract_text()
        mean, std = cv2.meanStdDev(queryImage)
        mean = [int(val) for val in mean]
        # queryImage[bbox[1]:bbox[3], bbox[0]:bbox[2]] = mean

        # Describe a 3D RGB histogram with 8 bins per channel
        desc = HistogramDescriptor((8, 8, 8))
        queryFeatures = desc.computeHSV(queryImage)

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

        # Append the final predicted list
        predicted.append(predicted_query)
        bounding_boxes.append(bbox)

# Save the results
with open("output" + ".pkl", "wb") as fp:
    pickle.dump(predicted, fp)
with open("bounding_boxes" + ".pkl", "wb") as fp:
    pickle.dump(bounding_boxes, fp)

# Evaluate the map accuracy
print("map@ {}: {}".format(1, evaluate(predicted, args["query1"] + "/gt_corresps.pkl", k=1)))
print("map@ {}: {}".format(5, evaluate(predicted, args["query1"] + "/gt_corresps.pkl", k=5)))

predicted = []
Results = []
bounding_boxes = []

for imagePath in sorted(list_images(args["query2"])):
    if "jpg" in imagePath:
        # Load the image
        image = cv2.imread(imagePath)

        th_open, stats = RemoveBackground.compute_removal_2(image)
        desc = HistogramDescriptor([8, 8, 8])
        pq = []
        bgb = []

        for i in range(0, len(stats)):
            bb = stats[i]
            image1 = image.copy()

            # cv2.imwrite("masks"+imagePath + str(i+1) + "_mask_res.png", th_open[bb[1]:bb[1] + bb[4], bb[0]:bb[0] + bb[2]])
            # cv2.imwrite("masks"+imagePath + str(i+1) + "_paint_res.png", image[bb[1]:bb[1] + bb[4], bb[0]:bb[0] + bb[2], :])

            text_id = RemoveText(image[bb[1]:bb[1] + bb[4], bb[0]:bb[0] + bb[2], :])
            bbox = text_id.extract_text()

            image_rec = cv2.rectangle(image[bb[1]:bb[1] + bb[4], bb[0]:bb[0] + bb[2], :], (bbox[0], bbox[1]),
                                      (bbox[2], bbox[3]), (0, 0, 255), 8)

            cv2.imshow("image", image[bb[1]:bb[1] + bb[4], bb[0]:bb[0] + bb[2], :])
            mean, std = cv2.meanStdDev(image1)
            # image[bb[1]:bb[1] + bb[4], bb[0]:bb[0] + bb[2]][bbox[1]:bbox[3], bbox[0]:bbox[2]] = 1
            queryFeatures = desc.computeHSV(image[bb[1]:bb[1] + bb[4], bb[0]:bb[0] + bb[2]])

            # Add the bounding box to list
            bbox = [bbox[0] + bb[0], bbox[1] + bb[1], bbox[2] + bb[0], bbox[3] + bb[1]]
            bgb.append(bbox)

            # Perform the search
            searcher = Searcher(index)
            results = searcher.search(queryFeatures)
            predicted_query = []

            # Loop over the top ten results
            for j in range(0, 10):
                # Grab the result
                (score, imageName) = results[j]
                predicted_query.append(int(imageName.replace(".jpg", "")))
                print(i, "\t{}. {} : {:.3f}".format(j + 1, imageName, score))

            # Append the final predicted list
            pq.append(predicted_query)

        predicted.append(pq)
        bounding_boxes.append(bgb)

# Save the results
with open("output1" + ".pkl", "wb") as fp:
    pickle.dump(predicted, fp)
with open("bounding_boxes1" + ".pkl", "wb") as fp:
    pickle.dump(bounding_boxes, fp)


# Evaluate the map accuracy
print("map@ {}: {}".format(1, evaluate(predicted, args["query2"] + "/gt_corresps.pkl", k=1)))
print("map@ {}: {}".format(5, evaluate(predicted, args["query2"] + "/gt_corresps.pkl", k=5)))
