# Import the necessary packages
import argparse
import collections
import enum
from msvcrt import kbhit
import pickle

import cv2
from imutils.paths import list_images
import os
from packages import RemoveText, RemoveBackground, HistogramDescriptor, RemoveNoise, TextureDescriptors, TextDescriptors, Searcher
from packages.average_precicion import mapk

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--query", default="../dataset/qst2_w3", help="Path to the image")
ap.add_argument("-a", "--augmented", default="y", help="augmented dataset / with noise?")
ap.add_argument("-c", "--color", default="y", help="Do we use color descriptors?")
ap.add_argument("-t", "--texture", default="y", help="Do we use texture descriptors?")
ap.add_argument("-txt", "--text", default="y", help="Do we use text descriptors?")
#ap.add_argument("-a", "--gt", default="y", help="there is a ground truth")
args = vars(ap.parse_args())

augmented = "y" == args["augmented"]
color = "y" == args["color"]
texture = "y" == args["texture"]
text = "y" == args["text"]

# Initialize a Dictionary to store our images and features
index_color = {}
index_texture = {}
index_text = {}
descriptor = HistogramDescriptor((8, 8, 8))
descriptor1 = TextureDescriptors()

def get_k_searcher(index, queryFeatures, k = 10):
    # Perform the search
    searcher = Searcher(index)
    results = searcher.search(queryFeatures)

    predicted_query = []
    # print("text", text)

    # Loop over the top ten results
    for j in range(0, k):
        # Grab the result
        (score, imageName) = results[j]
        predicted_query.append(int(imageName.replace(".jpg", "")))
        print("\t{}. {} : {:.3f}".format(j + 1, imageName, score))

    return predicted_query
# Initialize image descriptor

print("Indexing images")

if texture or color:
    # Use list_images to grab the image paths and loop over them
    for imagePath in sorted(list_images("../dataset/bbdd")):
        if "jpg" in imagePath:
            # Extract our unique image ID (i.e. the filename)
            path = imagePath[imagePath.rfind("_") + 1:]

            # Load the image, compute histogram and update the index
            image = cv2.imread(imagePath)
            if color:
                index_color[path] = descriptor.computeHSV(image)
            if texture:
                index_texture[path] = descriptor1.compute_hog(image)

if text:
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
index_color = collections.OrderedDict(sorted(index_color.items()))
index_texture = collections.OrderedDict(sorted(index_texture.items()))
index_text = index_text


predicted_color = []
predicted_texture = []
predicted_text = []
Results = []
bounding_boxes = []
text_readed_path = "./text_qst2_w3/"

for imagePath in sorted(list_images(args["query"])):
    if "jpg" in imagePath:
        if augmented and not "non_augmented" in imagePath or not augmented and "non_augmented" in imagePath:
            # Load the image
            print(imagePath)
            image = cv2.imread(imagePath)

            th_open, stats = RemoveBackground.compute_removal_2(image)
            #cv2.imwrite("masks/"+imagePath+".png", th_open)

        
            desc = TextureDescriptors()
        
            desc = HistogramDescriptor([8, 8, 8])
            pq_color = []
            pq_texture = []
            pq_text = []
            bgb = []
            txtr = []
            
            for i in range(0, len(stats)):
                bb = stats[i]
                image1 = image.copy()

                # cv2.imwrite(str(i) + "_mask_res.png", th_open[bb[1]:bb[1] + bb[4], bb[0]:bb[0] + bb[2]])
                # cv2.imwrite(str(i) + "_paint_res.png", image[bb[1]:bb[1] + bb[4], bb[0]:bb[0] + bb[2], :])
                
                text_id = RemoveText(image[bb[1]:bb[1] + bb[4], bb[0]:bb[0] + bb[2], :])
                bbox = text_id.extract_text()

                image_rec = cv2.rectangle(image[bb[1]:bb[1] + bb[4], bb[0]:bb[0] + bb[2], :], (bbox[0], bbox[1]),
                                        (bbox[2], bbox[3]), (0, 0, 255), 8)
                img_bb = image[bb[1]:bb[1] + bb[4], bb[0]:bb[0] + bb[2], :]
                if color:
                    desc_c = HistogramDescriptor((8, 8, 8))
                    queryFeatures_c = desc_c.computeHSV(img_bb) # color works better without removing noise
                    pq_color.append(get_k_searcher(index_color, queryFeatures_c))
                if texture:
                    desc_t = TextureDescriptors()
                    queryFeatures_t = desc_t.compute_hog(img_bb)
                    pq_texture.append(get_k_searcher(index_texture, queryFeatures_t))
                if text:
                    rn = RemoveNoise(img_bb)
                    queryImage_rn = rn.denoise_image()
                    td = TextDescriptors()
                    aux = td.get_k_images(queryImage_rn, index_text)
                    pq_text.append(aux[0])
                    txtr.append(aux[3]+"\n")
                # Perform the search
                bbox = [bbox[0]+bb[0], bbox[1]+bb[1], bbox[2]+bb[0], bbox[3]+bb[1]]
                bgb.append(bbox)


            predicted_color.append(pq_color)
            predicted_texture.append(pq_texture)
            predicted_text.append(pq_text)
            bounding_boxes.append(bgb)
            with open(text_readed_path+imagePath[-9:-4]+".txt", 'w') as f:
                for t in txtr:
                    f.write(t)

with open("output_color3a" + ".pkl", "wb") as fp:
    pickle.dump(predicted_color, fp)

with open("output_texture3a" + ".pkl", "wb") as fp:
    pickle.dump(predicted_texture, fp)

with open("output_text3a" + ".pkl", "wb") as fp:
    pickle.dump(predicted_text, fp)

with open("output_boundingbox3a" + ".pkl", "wb") as fp:
    pickle.dump(bounding_boxes, fp)


file = open("output_color3a.pkl", 'rb')
predicted_color = pickle.load(file)
file = open("output_texture3a.pkl", 'rb')
predicted_texture =  pickle.load(file)
file = open("output_text3a.pkl", 'rb')
predicted_text = pickle.load(file)

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

"""
# Evaluate the map accuracy
if color:
    print("Prediction of color:")
    print("map@ {}: {}".format(1, evaluate(predicted_color, args["query"] + "/gt_corresps.pkl", k=1)))
    print("map@ {}: {}".format(5, evaluate(predicted_color, args["query"] + "/gt_corresps.pkl", k=5)))

if texture:
    print("Prediction of texture:")
    print("map@ {}: {}".format(1, evaluate(predicted_texture, args["query"] + "/gt_corresps.pkl", k=1)))
    print("map@ {}: {}".format(5, evaluate(predicted_texture, args["query"] + "/gt_corresps.pkl", k=5)))

if text:
    print("Prediction of text:")
    print("map@ {}: {}".format(1, evaluate(predicted_text, args["query"] + "/gt_corresps.pkl", k=1)))
    print("map@ {}: {}".format(5, evaluate(predicted_text, args["query"] + "/gt_corresps.pkl", k=5)))

"""
def calculate_one_descriptor(voting, predicted, prob):

    for i, l in enumerate(predicted):
        l_v = voting[i]
        for j, p_picture in enumerate(l):
            v = l_v[j]
            for k, p in enumerate(p_picture):
                if not p in v:
                    v[p] = prob * (len(p_picture) - j)
                else:
                    v[p] =  v[p] + prob * (len(p_picture) - j)


def calculate_soft_voting(p_color, p_texture, p_text):

    voting = []
    longi = max(len(p_color), len(p_texture), len(p_text))
    for it in range(longi):
        v = []
        longi_2 = max(len(p_texture[it]), 1)
        for it2 in range(longi_2):
            v.append({})
        voting.append(v)
    calculate_one_descriptor(voting, p_color, 0.8)
    calculate_one_descriptor(voting, p_texture, 0.733)
    calculate_one_descriptor(voting, p_text, 0.266)

    voted = [[sorted(v.items(), key=lambda item: item[1], reverse = True) for v in l_v]for l_v in voting]

    return [[[p[0] for i, p in enumerate(predictions) if i < 10]for predictions in photo_list] for photo_list in voted]


predicted_all = calculate_soft_voting(predicted_color, predicted_texture, predicted_text)

with open("output_comb" + ".pkl", "wb") as fp:
    pickle.dump(predicted_all, fp)


"""
print("Prediction of a combination of all:")
print("map@ {}: {}".format(1, evaluate(predicted_all, args["query"] + "/gt_corresps.pkl", k=1)))
print("map@ {}: {}".format(5, evaluate(predicted_all, args["query"] + "/gt_corresps.pkl", k=5)))

"""


