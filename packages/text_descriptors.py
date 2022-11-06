from string import ascii_letters, digits
import cv2
import pytesseract
import operator
import os
import textdistance
from packages import RemoveText
import re

class TextDescriptors:
    # Define function to ocr text
    def read_text(self, queryImage, bbox, useBB):
        if os.name == "nt": # only run it in windows
            pytesseract.pytesseract.tesseract_cmd = "tesseract.exe"
        # Extract the coordinates
        x = bbox[0]
        y = bbox[1]
        w = bbox[2]
        h = bbox[3]

        if bbox != [0, 0, 0, 0]:
            if useBB:
                roi = queryImage[y:h, x:w]
            else:
                roi = queryImage
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            _, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            text = pytesseract.image_to_string(gray, config='--psm 6')

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
    def get_text_distance(self, text_1, text_2, distance_metric):
        # For metric Levensthein
        if distance_metric == "Levensthein":
            distance = textdistance.levenshtein(text_1, text_2)

        # For metric Hamming
        elif distance_metric == "Hamming":
            distance = textdistance.hamming(text_1, text_2)

        # For metric Damerau_levenshtein
        elif distance_metric == "Damerau_levenshtein":
            distance = textdistance.damerau_levenshtein(text_1, text_2)

        # For metric Jaccard
        elif distance_metric == "Jaccard":
            distance = textdistance.jaccard(text_1, text_2)

        # For metric Ratcliff
        elif distance_metric == "Ratcliff-Obershelp":
            distance = textdistance.ratcliff_obershelp(text_1, text_2)
        # For metric Arithmetic
        elif distance_metric == "Arithmetic":
            distance = textdistance.arith_ncd(text_1, text_2)
        # For metric Entropy
        elif distance_metric == "Entropy":
            distance = textdistance.entropy_ncd(text_1, text_2)


        # Print error if metric is not valid
        else:
            print('Metric doesn\'t exist')

        # Return the distance
        return distance
    
    def cleanstr(self, st):
        st = st.lower() + " "
        #st = re.sub(r'\b\w{1,4}\b.' , ' ', st)
        st = re.sub(r'[0-9]+' , ' ', st)
        st = re.sub(r'\b[aeiou][aeiou]*\b' , ' ', st) # palabras de vocales juntas
        st = re.sub(r'\b[a-z]*(aa|ee|ii|oo|uu)+[a-z]*\b' , ' ', st) # 
        st = re.sub(r'\b[bcdfghjklmnpqrstvwxz][bcdfghjklmnpqrstvwxz]*\b' , ' ', st)
        st = re.sub(r'\b(bcdfghjklmnpqrstvwxz)\b' , ' ', st)
        st = re.sub(r'[ ]+' , ' ', st)
        return st

    def cleanstrstrong(self, st):
        st = st.lower() + " "
        st = re.sub(r'\b\w{1,4}\b.' , ' ', st)
        st = re.sub(r'[0-9]+' , ' ', st)
        st = re.sub(r'\b[aeiou][aeiou]*\b' , ' ', st) # palabras de vocales juntas
        st = re.sub(r'\b[a-z]*(aa|ee|ii|oo|uu)+[a-z]*\b' , ' ', st) # 
        st = re.sub(r'\b[bcdfghjklmnpqrstvwxz][bcdfghjklmnpqrstvwxz]*\b' , ' ', st)
        st = re.sub(r'\b(bcdfghjklmnpqrstvwxz)\b' , ' ', st)
        st = re.sub(r'[ ]+' , ' ', st)
        return st

    def get_k_images(self, image, index, k=10, distance_metric="Levensthein"):
        # Get Bounding box of the image
        rt = RemoveText(image)
        bb = rt.extract_text()

        # Initialize distance
        distances = {}
        text = self.read_text(image, bb, True)
        text = self.cleanstr(text)
        print("Cleaned:", text)
        # Loop over the dataset texts
        for id, dataset_text in index.items():

            # Calculate similarity between the current text and dataset texts
            if dataset_text != 'empty':
                dataset_text = dataset_text.replace("(", "").replace("'", " ").replace(")", "")
                distances[id] = self.get_text_distance(text.lower(), dataset_text.split(",", 1)[0].strip(), distance_metric)

            else:
                distances[id] = 100

        # Calculate the minimum distance and get the top k images
        min_distance = min(distances.values())
        author_images = [key for key in distances if distances[key] == min_distance]
        k_predicted_images = (sorted(distances.items(), key=operator.itemgetter(1), reverse=False))[:k]

        if min_distance >= len(text.replace(" ","")):
            print("Recomputing text without BB")
            distances = {}
            text = self.read_text(image, bb, False)
            text = self.cleanstrstrong(text)
            print("Cleaned:", text)
            # Loop over the dataset texts
            for id, dataset_text in index.items():

                # Calculate similarity between the current text and dataset texts
                if dataset_text != 'empty':
                    dataset_text = dataset_text.replace("(", "").replace("'", " ").replace(")", "")
                    distances[id] = self.get_text_distance(text.lower(), dataset_text.split(",", 1)[0].strip(), distance_metric)

                else:
                    distances[id] = 100

            # Calculate the minimum distance and get the top k images
            min_distance = min(distances.values())
            author_images = [key for key in distances if distances[key] == min_distance]
            k_predicted_images = (sorted(distances.items(), key=operator.itemgetter(1), reverse=False))[:k]


        # Return the predictions
        return [int(predicted_image[0][:-4]) for predicted_image in k_predicted_images], author_images, distances, text
