# Import required packages
import argparse
import collections
import pickle
import time

import cv2
from imutils.paths import list_images
from imutils.feature.factories import FeatureDetector_create, DescriptorExtractor_create, DescriptorMatcher_create
from packages import HistogramDescriptor, RemoveText, Searcher, RGBHistogram, RemoveBackground, DetectAndDescribe, \
    SearchFeatures, RemoveNoise, extract_angle, TextDescriptors
from packages.average_precicion import mapk
import os

def flatten(l):
    return [[item] for sublist in l for item in sublist]




def balance_lists(gt, pred):
    """
    Only works for a difference of 1 in each iteration
    """
    gt_res = []
    pred_res = []

    for g, p in zip(gt, pred):
        add_g = g
        add_p = p
        if len(g) > len(p):
            aux = 10 * [-1]
            add_p.append(aux)
        elif len(p) > len(g):
            aux = [-1]
            add_g += aux
        pred_res += add_p
        gt_res.append(add_g)

    gt_res = flatten(gt_res)
    return gt_res, pred_res


def evaluate(predicted, ground_truth, k):
    file = open(ground_truth, 'rb')
    actual = pickle.load(file)

    actual, predicted = balance_lists(actual, predicted)

    result = mapk(actual=actual, predicted=predicted, k=k)
    return result

def evaluate_2(predicted, ground_truth):
    file = open(ground_truth, 'rb')
    actual = pickle.load(file)

    actual, predicted = balance_lists(actual, predicted)
    summation = 0
    count = 0
    txd = TextDescriptors()
    for i in range(len(actual)):
        summation += txd.get_text_distance(actual[i], predicted[i], distance_metric="Levensthein")
        count += 1
    return summation/count



# Construct argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--index", default="../dataset/bbdd", help="Path to the image dataset")
ap.add_argument("-q1", "--query1", default="../dataset/qst1_w5", help="Path to the query image")
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
orb = cv2.ORB_create(10000)

# initialize the feature extractor
extractor = DescriptorExtractor_create(args["extractor"])
index_text = {}

"""index = {}

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
        kps = orb.detect(image, None)
        print("length kps", len(kps))

        # extract features from each of the keypoint regions in the images
        if len(kps) != 0:
            (kps, features) = orb.compute(image, kps)
            print("length features", len(features))
            index[path] = features

# Sort the dictionary according to the keys
index = collections.OrderedDict(sorted(index.items()))

with open("index_orb" + ".pkl", "wb") as fp:
    pickle.dump(index, fp)
    


file = open("index_orb.pkl", 'rb')
index = pickle.load(file)  
    
"""


# Initialize a Dictionary to store texts
all_files = sorted(os.listdir("../dataset/bbdd"))
# Use list_images to grab the image paths and loop over them
for imagePath in all_files:
    if "txt" in imagePath:
        # Extract our unique image ID (i.e. the filename)
        path = imagePath[imagePath.rfind("_") + 1:]

        # Open the text file and read contents
        file = open("../dataset/bbdd"+'/'+imagePath, "r")
        line = file.readline()
        if line.strip():
            text = line.lower().replace("(", "").replace("'", " ").replace(")", "")
        else:
            text = 'empty'

        # Add the text to list of dictionaries
        index_text[path] = text






# Sort the dictionary according to the keys
predicted = []
Results = []
bounding_boxes = []
predicted_text = []
text_read = []

start_time = time.time()
# initialize the feature detector
orb = cv2.ORB_create(10000)

# initialize the keypoint matcher
matcher = DescriptorMatcher_create(args["matcher"])

results = {}
index_results = {}
text_readed_path = "./text_qst1_w5/"

# Load the query images
for imagePath1 in sorted(list_images(args["query1"])):
    if "jpg" in imagePath1 and "non_augmented" in imagePath1:
        print(imagePath1)
        image = cv2.imread(imagePath1)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        noise = RemoveNoise(image)
        image = noise.denoise_image()
        angle, image = extract_angle(image)

        th_open, stats = RemoveBackground.compute_removal(image)
        pq = []
        txtdist = []
        pq_text = []
        txtr = []
        for i in range(0, len(stats)):
            bb = stats[i]
            img_bb = image[bb[1]:bb[1] + bb[4], bb[0]:bb[0] + bb[2], :]


            """
            img_bb = cv2.cvtColor(img_bb, cv2.COLOR_BGR2GRAY)

            # detect keyPoints in the two images
            kps = orb.detect(img_bb, None)
            print(len(kps))

            # extract features from each of the keypoint regions in the images
            (kps, features) = orb.compute(img_bb, kps)

            # Perform the search
            searcher = SearchFeatures(index)
            results = searcher.search(features)
            predicted_query = []
            Score = []

            # Loop over the top ten results
            for j in range(0, 10):
                # Grab the result
                (score, imageName) = results[j]
                Score.append(score)
                predicted_query.append(int(imageName.replace(".jpg", "")))
                print("\t{}. {} : {:.3f}".format(j + 1, imageName, score))

            Score = sorted(Score, reverse=True)
            if Score[0] < 3*len(kps)/100:
                predicted_query = [-1]

            pq.append(predicted_query)
            print(pq)
            """

            rn = RemoveNoise(img_bb)
            queryImage_rn = rn.denoise_image()
            td = TextDescriptors()
            aux = td.get_k_images(queryImage_rn, index_text)
            pq_text.append(aux[0])
            txtr.append(aux[3]+"\n")


        # Append the final predicted list
        predicted.append(pq)
        predicted_text.append(pq_text)
        text_read.append(txtr)

        with open(text_readed_path+imagePath[-9:-4]+".txt", 'w') as f:
            for t in txtr:
                f.write(t)
        #print(predicted)


"""# Save the results
with open("result" + ".pkl", "wb") as fp:
    pickle.dump(predicted, fp)
with open("bounding_boxes_2" + ".pkl", "wb") as fp:
    pickle.dump(bounding_boxes, fp)

file = open("result.pkl", 'rb')
predicted = pickle.load(file)
file = open("bounding_boxes_2.pkl", 'rb')
bounding_boxes = pickle.load(file)

"""
with open("predicted_text" + ".pkl", "wb") as fp:
    pickle.dump(predicted_text, fp)
with open("text_read" + ".pkl", "wb") as fp:
    pickle.dump(text_read, fp)

file = open("predicted_text.pkl", 'rb')
predicted_text = pickle.load(file)
file = open("text_read.pkl", 'rb')
text_read = pickle.load(file)
"""
# Evaluate the map accuracy
print("map@ {}: {}".format(1, evaluate(predicted, args["query1"] + "/gt_corresps.pkl", k=1)))
print("map@ {}: {}".format(5, evaluate(predicted, args["query1"] + "/gt_corresps.pkl", k=5)))
print("--- %s seconds ---" % (time.time() - start_time))
"""

print("Prediction of text:")
print("map@ {}: {}".format(1, evaluate(predicted_text, args["query1"] + "/gt_corresps.pkl", k=1)))
print("map@ {}: {}".format(5, evaluate(predicted_text, args["query1"] + "/gt_corresps.pkl", k=5)))
print("Avg text distance:")
print("{}".format(evaluate_2(text_read, args["query1"] + "/gt_corresps.pkl")))
