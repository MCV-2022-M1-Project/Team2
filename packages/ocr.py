from string import ascii_letters, digits
import cv2
import pytesseract
from PIL import Image
import operator
import jellyfish
import argparse
from packages import RemoveText, RemoveBackground

# Construct argument parser and parse arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--index", default="../dataset/bbdd", help="Path to the dataset")
ap.add_argument("-q", "--query1", default="../dataset/qsd1_w2", help="Path to the query dataset")
args = vars(ap.parse_args())


# Define function to ocr text
def read_text(queryImage, bbox):
    # Extract the coordinates
    x = bbox[0]
    y = bbox[1]
    w = bbox[2]
    h = bbox[3]

    if bbox != [0, 0, 0, 0]:
        roi = queryImage[y:h, x:w]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        text = pytesseract.image_to_string(gray)

    else:
        text = "not found"

    # Filter the text
    special_chars = [c for c in set(text).difference(ascii_letters + digits + ' ')]
    text_filtered = ''.join((filter(lambda s: s not in special_chars, text)))
    final_text = ' '.join(text_filtered.split())

    # Return the text
    print("Text: ", final_text)
    return final_text


# Define function to Compare similarities
def get_text_distance(text_1, text_2, distance_metric):
    # For metric Levensthein
    if distance_metric == "Levensthein":
        distance = jellyfish.levenshtein_distance(text_1, text_2)

    # For metric Hamming
    elif distance_metric == "Hamming":
        distance = jellyfish.hamming_distance(text_1, text_2)

    # For metric Hamming
    elif distance_metric == "Hamming":
        distance = jellyfish.damerau_levenshtein_distance(text_1, text_2)

    # Print error if metric is not valid
    else:
        print('Metric doesn\'t exist')

    # Return the distance
    return distance
