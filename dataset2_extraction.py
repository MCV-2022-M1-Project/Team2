# Import the necessary packages
import argparse
import collections
import pickle

import cv2
from imutils.paths import list_images

from packages import RemoveText, RemoveBackground, HistogramDescriptor, Searcher, TextureDescriptors
from packages.average_precicion import mapk

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--query", default="../dataset/qsd2_w3", help="Path to the image")
ap.add_argument("-t", "--texture", default="y", help="Do we use texture?")
ap.add_argument("-a", "--augmented", default="y", help="augmented dataset?")
ap.add_argument("-a", "--gt", default="y", help="there is a ground truth")
args = vars(ap.parse_args())

texture = "y" == args["texture"]
augmented = "y" == args["augmented"]


# Initialize a Dictionary to store our images and features
index = {}
descriptor = None

# Initialize image descriptor
if texture:
    descriptor = TextureDescriptors()
else:
    descriptor = HistogramDescriptor([8, 8, 8])
print("Indexing images")

# Use list_images to grab the image paths and loop over them
for imagePath in sorted(list_images("../dataset/bbdd")):
    if "jpg" in imagePath:
        # Extract our unique image ID (i.e. the filename)
        path = imagePath[imagePath.rfind("_") + 1:]

        # Load the image, compute histogram and update the index
        image = cv2.imread(imagePath)
        if texture:
            features = descriptor.compute_hog(image)
        else:
            features = descriptor.computeHSV(image)
        index[path] = features

# Sort the dictionary according to the keys
index = collections.OrderedDict(sorted(index.items()))
predicted = []
Results = []
bounfing_boxes = []
desc = None

for imagePath in sorted(list_images(args["query"])):
    if "jpg" in imagePath:
        if augmented and not "non_augmented" in imagePath or not augmented and "non_augmented" in imagePath:
            # Load the image
            print(imagePath)
            image = cv2.imread(imagePath)

            th_open, stats = RemoveBackground.compute_removal_2(image)
            cv2.imwrite("masks/"+imagePath+".png", th_open)

            if texture:
                desc = TextureDescriptors()
            else:
                desc = HistogramDescriptor([8, 8, 8])
            pq = []
            bgb = []

            for i in range(0, len(stats)):
                bb = stats[i]
                image1 = image.copy()

                # cv2.imwrite(str(i) + "_mask_res.png", th_open[bb[1]:bb[1] + bb[4], bb[0]:bb[0] + bb[2]])
                # cv2.imwrite(str(i) + "_paint_res.png", image[bb[1]:bb[1] + bb[4], bb[0]:bb[0] + bb[2], :])

                text_id = RemoveText(image[bb[1]:bb[1] + bb[4], bb[0]:bb[0] + bb[2], :])
                bbox = text_id.extract_text()

                image_rec = cv2.rectangle(image[bb[1]:bb[1] + bb[4], bb[0]:bb[0] + bb[2], :], (bbox[0], bbox[1]),
                                        (bbox[2], bbox[3]), (0, 0, 255), 8)

                #cv2.imshow("image", image[bb[1]:bb[1] + bb[4], bb[0]:bb[0] + bb[2]])
                #cv2.waitKey(0)
                #mean, std = cv2.meanStdDev(image1)
                # image[bb[1]:bb[1] + bb[4], bb[0]:bb[0] + bb[2]][bbox[1]:bbox[3], bbox[0]:bbox[2]] = 1
                if texture:
                    queryFeatures = desc.compute_hog(image[bb[1]:bb[1] + bb[4], bb[0]:bb[0] + bb[2], :])
                else:
                    queryFeatures = desc.computeHSV(image[bb[1]:bb[1] + bb[4], bb[0]:bb[0] + bb[2], :])

                # Perform the search
                searcher = Searcher(index)
                results = searcher.search(queryFeatures)
                predicted_query = []
                bbox = [bbox[0]+bb[0], bbox[1]+bb[1], bbox[2]+bb[0], bbox[3]+bb[1]]
                bgb.append(bbox)

                # Loop over the top ten results
                for j in range(0, 10):
                    # Grab the result
                    (score, imageName) = results[j]
                    predicted_query.append(int(imageName.replace(".jpg", "")))
                    print("\t{}. {} : {:.3f}".format(j + 1, imageName, score))

                # Append the final predicted list
                pq.append(predicted_query)


            predicted.append(pq)
            bounfing_boxes.append(bgb)

# Save the results
with open("output2_2" + ".pkl", "wb") as fp:
    pickle.dump(predicted, fp)
with open("bounding_boxes2_2" + ".pkl", "wb") as fp:
    pickle.dump(bounfing_boxes, fp)



def flatten(l):
    return [[item] for sublist in l for item in sublist]

def balance_lists(gt, pred):
    """
    Only works for a difference of 1 in each iteration
    """
    gt_res = []
    pred_res = []

    for g,p in zip(gt,pred):
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
    return (gt_res, pred_res)


# Evaluate the predictions
def evaluate(predicted, ground_truth, k):
    file = open(ground_truth, 'rb')
    actual = pickle.load(file)

    actual, predicted = balance_lists(actual, predicted)

    result = mapk(actual=actual, predicted=predicted, k=k)
    return result
#file = open("output2_2.pkl", 'rb')
#predicted = pickle.load(file)

# Evaluate the map accuracy
print("map@ {}: {}".format(1, evaluate(predicted, args["query"] + "/gt_corresps.pkl", k=1)))
print("map@ {}: {}".format(5, evaluate(predicted, args["query"] + "/gt_corresps.pkl", k=5)))
