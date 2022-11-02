# Import required packages
import argparse
import collections
import pickle
import cv2
from imutils.paths import list_images
from imutils.feature import FeatureDetector_create, DescriptorExtractor_create
from packages import HistogramDescriptor, RemoveText, Searcher, RGBHistogram, RemoveBackground, DetectAndDescribe
from packages.average_precicion import mapk


def evaluate(predicted, ground_truth, k):
    file = open(ground_truth, 'rb')
    actual = pickle.load(file)
    result = mapk(actual=actual, predicted=predicted, k=k)
    return result


# Construct argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--index", default="../dataset/bbdd", help="Path to the image dataset")
ap.add_argument("-q1", "--query1", default="../dataset/qsd2_w1", help="Path to the query image")
ap.add_argument("-q2", "--query2", default="../dataset/qsd2_w2", help="Path to the query image")

args = vars(ap.parse_args())

# Initialize a Dictionary to store our images and features
index = {}

# Initialize the keypoint detector, local invariant descriptor, and the descriptor pipeline
detector = FeatureDetector_create("SURF")
descriptor = DescriptorExtractor_create("RootSIFT")
detectAndDescribe = DetectAndDescribe(detector, descriptor)
print("Indexing images")

# Use list_images to grab the image paths and loop over them
for imagePath in list_images(args["index"]):
    if "jpg" in imagePath:
        # Extract our unique image ID (i.e. the filename)
        path = imagePath[imagePath.rfind("_") + 1:]

        # Load the image, compute features and update the index
        image = cv2.imread(imagePath)
        (kps, descriptors) = detectAndDescribe.describe(image)
        index[path] = [kps, descriptors]

# Sort the dictionary according to the keys
index = collections.OrderedDict(sorted(index.items()))
predicted = []
Results = []
bounding_boxes = []

# Load the query images
for imagePath1 in sorted(list_images(args["query1"])):
    if "jpg" in imagePath1:
        queryImage = cv2.imread(imagePath1)
        print("query: {}".format(imagePath1))

        # text_id = RemoveText(queryImage)
        # bbox = text_id.extract_text()
        # mean, std = cv2.meanStdDev(queryImage)
        # mean = [int(val) for val in mean]
        # queryImage[bbox[1]:bbox[3], bbox[0]:bbox[2]] = mean

        # Describe a 3D RGB histogram with 8 bins per channel
        queryKps, queryDescriptors = detectAndDescribe.describe(queryImage)

        # Perform the search
        searcher = Searcher(index)
        results = searcher.search(queryDescriptors)
        predicted_query = []

        # Loop over the top ten results
        for j in range(0, 10):
            # Grab the result
            (score, imageName) = results[j]
            predicted_query.append(int(imageName.replace(".jpg", "")))
            print("\t{}. {} : {:.3f}".format(j + 1, imageName, score))

        # Append the final predicted list
        predicted.append(predicted_query)
        # bounding_boxes.append(bbox)

# Evaluate the map accuracy
print("map@ {}: {}".format(1, evaluate(predicted, args["query1"] + "/gt_corresps.pkl", k=1)))
print("map@ {}: {}".format(5, evaluate(predicted, args["query1"] + "/gt_corresps.pkl", k=5)))

# Save the results
with open("output_2" + ".pkl", "wb") as fp:
    pickle.dump(predicted, fp)
with open("bounding_boxes_2" + ".pkl", "wb") as fp:
    pickle.dump(bounding_boxes, fp)