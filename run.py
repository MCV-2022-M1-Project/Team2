# Import the necessary packages
import argparse
import collections
import pickle
from xmlrpc.client import Boolean

import cv2
from imutils.paths import list_images

from packages import RemoveText, RemoveBackground, HistogramDescriptor, Searcher
from packages.average_precicion import mapk, apk


def indexBBDD():
    # Initialize a Dictionary to store our images and features
    index = {}

    # Initialize image descriptor
    descriptor = HistogramDescriptor((8, 8, 8))
    print("Indexing images")

    # Use list_images to grab the image paths and loop over them
    for imagePath in list_images("../dataset/bbdd"):
        if "jpg" in imagePath:
            # Extract our unique image ID (i.e. the filename)
            path = imagePath[imagePath.rfind("_") + 1:]

            # Load the image, compute histogram and update the index
            image = cv2.imread(imagePath)
            features = descriptor.computeHSV(image)
            index[path] = features
    return index






# Evaluate the predictions
def evaluate(predicted, ground_truth, k, multiple):
    file = open(ground_truth, 'rb')
    actual = pickle.load(file)
    result = mapk(actual=actual, predicted=predicted, k=k)

    return result


def qs1_w2(gt, index):
    # Sort the dictionary according to the keys
    index = collections.OrderedDict(sorted(index.items()))
    predicted = []
    Results = []
    folder = "../dataset/qsd1_w2"

    for imagePath in sorted(list_images(folder)):
        if "jpg" in imagePath:
            # Load the image
            image = cv2.imread(imagePath)

            desc = HistogramDescriptor([8, 8, 8])
            pq = []

            #image1 = image.copy()
            text_id = RemoveText(image)
            bbox = text_id.extract_text()

            #image_rec = cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 8)

            #mean, std = cv2.meanStdDev(image1)

            image[bbox[1]:bbox[3], bbox[0]:bbox[2]] = 1
            print(imagePath[-9:-4])
            queryFeatures = desc.computeHSV(image)

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
            pq.append(predicted_query)

        predicted.append(pq)

    # Save the results
    with open("output" + ".pkl", "wb") as fp:
        pickle.dump(predicted, fp)

    # Evaluate the map accuracy
    print("map@ {}: {}".format(1, evaluate(predicted, folder + "/gt_corresps.pkl", 1)))
    print("map@ {}: {}".format(5, evaluate(predicted, folder + "/gt_corresps.pkl", 5)))

def qs2_w2(gt, index):

    # Sort the dictionary according to the keys
    index = collections.OrderedDict(sorted(index.items()))
    predicted = []
    Results = []
    folder = "../dataset/qsd2_w2"
    for imagePath in sorted(list_images(folder)):
        if "jpg" in imagePath:
            # Load the image
            image = cv2.imread(imagePath)

            th_open, stats = RemoveBackground.compute_removal_2(image)
            desc = HistogramDescriptor([8, 8, 8])
            pq = []

            for i in range(0, len(stats)):
                bb = stats[i]
                image1 = image.copy()

                # cv2.imwrite(str(i) + "_mask_res.png", th_open[bb[1]:bb[1] + bb[4], bb[0]:bb[0] + bb[2]])
                # cv2.imwrite(str(i) + "_paint_res.png", image[bb[1]:bb[1] + bb[4], bb[0]:bb[0] + bb[2], :])

                text_id = RemoveText(image[bb[1]:bb[1] + bb[4], bb[0]:bb[0] + bb[2], :])
                bbox = text_id.extract_text()

                image_rec = cv2.rectangle(image[bb[1]:bb[1] + bb[4], bb[0]:bb[0] + bb[2], :], (bbox[0], bbox[1]),
                                        (bbox[2], bbox[3]), (0, 0, 255), 8)

                #cv2.imshow("image", image[bb[1]:bb[1] + bb[4], bb[0]:bb[0] + bb[2], :])
                mean, std = cv2.meanStdDev(image1)
                image[bb[1]:bb[1] + bb[4], bb[0]:bb[0] + bb[2]][bbox[1]:bbox[3], bbox[0]:bbox[2]] = 1
                queryFeatures = desc.computeHSV(image[bb[1]:bb[1] + bb[4], bb[0]:bb[0] + bb[2]])

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

    # Save the results
    with open("output" + ".pkl", "wb") as fp:
        pickle.dump(predicted, fp)
    # Evaluate the map accuracy
    print("map@ {}: {}".format(1, evaluate(predicted, folder + "/gt_corresps.pkl", 1)))
    print("map@ {}: {}".format(5, evaluate(predicted, folder + "/gt_corresps.pkl", 5)))

def selectDataset(name, gt, index):
    switcher = {
        "1w2": qs1_w2,
        "2w2": qs2_w2,
    }

    func = switcher.get(name, lambda: "Invalid dataset")
    # Execute the function
    func(gt, index)



def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def main():

    #ap = argparse.ArgumentParser()
    #ap.add_argument("-i", "--query", default="2w2", required=True, help="Name of the dataset to run")
    #ap.add_argument("-gt", "--ground_truth", required=True, type=str2bool, help="Does the dataset have ground truth?")
    #args = vars(ap.parse_args())

    index = indexBBDD()
    selectDataset("1w2", True, index)



if __name__ == "__main__":
    main()



