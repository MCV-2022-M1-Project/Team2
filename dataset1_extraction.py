# Import required packages
import argparse
import collections
import pickle
import cv2
from imutils.paths import list_images
from imutils.feature.factories import FeatureDetector_create, DescriptorExtractor_create, DescriptorMatcher_create
from packages import HistogramDescriptor, RemoveText, Searcher, RGBHistogram, RemoveBackground, DetectAndDescribe, \
    SearchFeatures
from packages.average_precicion import mapk


def evaluate(predicted, ground_truth, k):
    file = open(ground_truth, 'rb')
    actual = pickle.load(file)
    result = mapk(actual=actual, predicted=predicted, k=k)
    return result


# Construct argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--index", default="../dataset/bbdd", help="Path to the image dataset")
ap.add_argument("-q1", "--query1", default="../dataset/qsd1_w2", help="Path to the query image")
ap.add_argument("-q2", "--query2", default="../dataset/qsd2_w2", help="Path to the query image")
ap.add_argument("-d", "--detector", type=str, default="SURF",
                help="Kepyoint detector to use. "
                     "Options ['BRISK', 'DENSE', 'DOG', 'SIFT', 'FAST', 'FASTHESSIAN', 'SURF', 'GFTT', 'HARRIS', "
                     "'MSER', 'ORB', 'STAR']")
ap.add_argument("-e", "--extractor", type=str, default="SIFT",
                help="Feature extractor to use. Options ['RootSIFT', 'SIFT', 'SURF']")
ap.add_argument("-m", "--matcher", type=str, default="BruteForce",
                help="Feature matcher to use. Options ['BruteForce', 'BruteForce-SL2', 'BruteForce-L1', 'FlannBased']")

args = vars(ap.parse_args())

# initialize the feature detector
# if the user entered detector as "DOG" or "FASTHESSIAN", use the appropriate value
if args["detector"] == "DOG":
    detector = FeatureDetector_create("SIFT")
elif args["detector"] == "FASTHESSIAN":
    detector = FeatureDetector_create("SURF")
else:
    detector = FeatureDetector_create(args["detector"])

# initialize the feature extractor
extractor = DescriptorExtractor_create(args["extractor"])

"""
index = {}

print("[INFO] indexing")
for imagePath in sorted(list_images("../dataset/bbdd")):
    if "jpg" in imagePath:
        # Extract our unique image ID (i.e. the filename)
        path = imagePath[imagePath.rfind("_") + 1:]
        print("for", path)

        # Load the image, compute histogram and update the index
        image = cv2.imread(imagePath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # detect keypoints in the two images
        kps = detector.detect(image)
        print("length kps", len(kps))

        # extract features from each of the keypoint regions in the images
        (kps, features) = extractor.compute(image, kps)
        print("length features", len(features))
        index[path] = features

# Sort the dictionary according to the keys
index = collections.OrderedDict(sorted(index.items()))

with open("index" + ".pkl", "wb") as fp:
    pickle.dump(index, fp)"""
file = open("index.pkl", 'rb')
index = pickle.load(file)

# Sort the dictionary according to the keys
predicted = []
Results = []
bounding_boxes = []

if args["detector"] == "DOG":
    detector = FeatureDetector_create("SIFT")
elif args["detector"] == "FASTHESSIAN":
    detector = FeatureDetector_create("SURF")
else:
    detector = FeatureDetector_create(args["detector"])

# initialize the feature extractor
extractor = DescriptorExtractor_create(args["extractor"])

# initialize the keypoint matcher
matcher = DescriptorMatcher_create(args["matcher"])

results = {}
# Load the query images
for imagePath1 in sorted(list_images(args["query1"])):
    if "jpg" in imagePath1:
        print(imagePath1)
        image = cv2.imread(imagePath1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # detect keypoints in the two images
        kps = detector.detect(image)

        # extract features from each of the keypoint regions in the images
        (kps, features) = extractor.compute(image, kps)

        # Perform the search
        searcher = SearchFeatures(index)
        results = searcher.search(features)
        predicted_query = []

        # Loop over the top ten results
        for j in range(0, 10):
            # Grab the result
            (score, imageName) = results[j]
            predicted_query.append(int(imageName.replace(".jpg", "")))
            print("\t{}. {} : {:.3f}".format(j + 1, imageName, score))

        # Append the final predicted list
        print(predicted)
        predicted.append(predicted_query)

# Evaluate the map accuracy
print("map@ {}: {}".format(1, evaluate(predicted, args["query1"] + "/gt_corresps.pkl", k=1)))
print("map@ {}: {}".format(5, evaluate(predicted, args["query1"] + "/gt_corresps.pkl", k=5)))

# Save the results
with open("output_2" + ".pkl", "wb") as fp:
    pickle.dump(predicted, fp)
"""with open("bounding_boxes_2" + ".pkl", "wb") as fp:
    pickle.dump(bounding_boxes, fp)"""
