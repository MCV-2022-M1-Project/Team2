import numpy as np
from PIL import Image
import cv2
class RemoveBackground:
    @staticmethod
    def compute_removal(data):

        threshold = 65
        threshold_2 = 40
        
        row, col, _ = data.shape

        threshold_rec_col = round(col/1.5)
        threshold_rec_row = round(row/1.5)
        threshold_erase_col = round(col/3)
        threshold_erase_row = round(row/3)

        # Get average color of background taking a sample from each side of the picture
        valor = data[0, round(col/2), :].astype(int)
        valor = valor + data[row-1, round(col/2), :].astype(int)
        valor = valor + data[round(row/2), 0 , :].astype(int)
        valor = valor + data[round(row/2), col-1, :].astype(int)
        valor = valor / 4
        col_shadow = valor-120

        # Every pixel similar to the background color gets to False
        mask = np.logical_not((data[:,:,0] > valor[0] - threshold) & (data[:,:,1] > valor[1]  - threshold) & (data[:,:,2] > valor[2]  - threshold))
        # Every pixel similar to the background color with shadow gets to False
        mask = np.logical_and(mask, np.logical_not((data[:,:,0] > col_shadow[0] - threshold_2+10) & (data[:,:,0] < col_shadow[0]  + threshold_2) & (data[:,:,1] > col_shadow[1]  - threshold_2+10) & (data[:,:,1] < col_shadow[1]  + threshold_2) & (data[:,:,2] > col_shadow[2]  - threshold_2+10) & (data[:,:,2] < col_shadow[2] + threshold_2)))
        
        # mask = np.logical_not((data[:,:,0] > valor[0] - threshold) & (data[:,:,0] < valor[0]  + threshold) & (data[:,:,1] > valor[1]  - threshold) & (data[:,:,1] < valor[1]  + threshold) & (data[:,:,2] > valor[2]  - threshold) & (data[:,:,2] < valor[2] + threshold))
       
        # Reconstruct painting by rows
        for i in range(row):
            first = 0
            final = col
            found_first = False

            for j in range(col):
                if mask[i,j] > 0 and not found_first:
                    found_first = True
                    first = j
                if mask[i,j] > 0 and j - final <= threshold_rec_col:
                    final = j
            if found_first:
                mask[i,first:final] = 1
            
        # Reconstruct painting by columns
        for j in range(col):
            first = 0
            final = row
            found_first = False

            for i in range(row):
                if mask[i,j] > 0 and not found_first:
                    found_first = True
                    first = i
                if mask[i,j] > 0 and i - final <= threshold_rec_row:
                    final = i
            if found_first:
                mask[first:final,j] = 1

        ######
        # Erase small objects by columns
        for j in range(col):
            first = 0
            final = 0
            conti = False

            for i in range(row):
                if mask[i,j] > 0 and not conti:
                    conti = True
                    first = i
                elif conti and (i == row-1 or mask[i+1,j] <= 0):
                    final = i+1
                    conti = False
                    if final - first <= threshold_erase_row:
                        mask[first:final,j] = 0

        # Erase small objects by rows
        for i in range(row):
            first = 0
            final = 0
            conti = False

            for j in range(col):
                if mask[i,j] > 0 and not conti:
                    conti = True
                    first = j
                elif conti and (j == col-1 or mask[i,j+1] <= 0):
                    final = j+1
                    conti = False
                    if final - first <= threshold_erase_col:
                        mask[i,first:final] = 0
            
        # Erase again small objects by columns
        for j in range(col):
            first = 0
            final = 0
            conti = False

            for i in range(row):
                if mask[i,j] > 0 and not conti:
                    conti = True
                    first = i
                elif conti and (i == row-1 or mask[i+1,j] <= 0):
                    final = i+1
                    conti = False
                    if final - first <= threshold_erase_row:
                        mask[first:final,j] = 0
  
        return mask.astype("uint8")

    @staticmethod
    def compute_removal_2(data):
        threshold = 40
        threshold_2 = 30

        # Convert image to PIL image
        img = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(img)

        im = image.convert('RGBA')
        data = np.array(im)

        # Get the color histogram of the image
        histogram = image.histogram()
       
        # Take only the Red counts
        l1 = histogram[0:256]

        # Take only the Blue counts
        l2 = histogram[256:512]

        # Take only the Green counts
        l3 = histogram[512:768]


        th = 10 # Window to compare peak
        # Search peak for red
        topR = 230
        for i in range(230,0,-1):
            if l1[i] >= l1[topR]:
                topR = i
            elif topR-i > th:
                break

        # Search peak for green
        topG = 230
        for i in range(230,0,-1):
            if l2[i] >= l2[topG]:
                topG = i
            elif topG-i > th:
                break

        # Search peak for blue
        topB = 230
        for i in range(230,0,-1):
            if l3[i] >= l3[topB]:
                topB = i
            elif topB-i > th:
                break

        #########################
        # Search for first minimum after peak in red channel
        th = 10
        botR = topB
        for i in range(topB,0,-1):
            if l1[i] <= l1[botR]:
                botR = i
            elif botR-i > th:
                break

        # Search for first minimum after peak in green channel
        botG = topG
        for i in range(topB,0,-1):
            if l2[i] <= l2[botG]:
                botG = i
            elif botG-i > th:
                break

        # Search for first minimum after peak in blue channel
        botB = topB
        for i in range(topB,0,-1):
            if l3[i] <= l3[botB]:
                botB = i
            elif botB-i > th:
                break

        botR_shadow = botR - 120
        botG_shadow = botG - 120
        botB_shadow = botB - 120

        mask = np.logical_not((data[:,:,0] > botR - threshold) & (data[:,:,1] > botG  - threshold) & (data[:,:,2] > botB  - threshold))
        mask = np.logical_and(mask, np.logical_not((data[:,:,0] > botR_shadow - threshold_2+10) & (data[:,:,0] < botR_shadow  + threshold_2) & (data[:,:,1] > botG_shadow  - threshold_2+10) & (data[:,:,1] < botG_shadow  + threshold_2) & (data[:,:,2] > botB_shadow  - threshold_2+10) & (data[:,:,2] < botB_shadow + threshold_2)))
        
        return mask.astype("uint8")
