import numpy as np
from PIL import Image
import cv2
from math import ceil

class RemoveBackground_2:
    @staticmethod
    def compute_removal(data):
        # Define thresholds
        threshold = 65
        threshold_2 = 40

        # Grab the shape of the image
        row, col, _ = data.shape

        # Define the thresholds
        threshold_rec_col = round(col / 1.2)
        threshold_rec_row = round(row / 1.2)
        threshold_erase_col = round(col / 3)
        threshold_erase_row = round(row / 3)

        # Get average color of background taking a sample from each side of the picture
        valor = data[0, round(col / 2), :].astype(int)
        valor = valor + data[row - 1, round(col / 2), :].astype(int)
        valor = valor + data[round(row / 2), 0, :].astype(int)
        valor = valor + data[round(row / 2), col - 1, :].astype(int)
        valor = valor / 4
        col_shadow = valor - 120

        # Every pixel similar to the background color gets to False
        mask = np.logical_not((data[:, :, 0] > valor[0] - threshold) & (data[:, :, 1] > valor[1] - threshold) & (
                    data[:, :, 2] > valor[2] - threshold))

        # Every pixel similar to the background color with shadow gets to False
        mask = np.logical_and(mask, np.logical_not(
            (data[:, :, 0] > col_shadow[0] - threshold_2 + 10) & (data[:, :, 0] < col_shadow[0] + threshold_2) & (
                        data[:, :, 1] > col_shadow[1] - threshold_2 + 10) & (
                        data[:, :, 1] < col_shadow[1] + threshold_2) & (
                        data[:, :, 2] > col_shadow[2] - threshold_2 + 10) & (
                        data[:, :, 2] < col_shadow[2] + threshold_2)))

        # mask = np.logical_not((data[:,:,0] > valor[0] - threshold) & (data[:,:,0] < valor[0]  + threshold) & (data[
        # :,:,1] > valor[1]  - threshold) & (data[:,:,1] < valor[1]  + threshold) & (data[:,:,2] > valor[2]  -
        # threshold) & (data[:,:,2] < valor[2] + threshold))

        # Reconstruct painting by rows

        for i in range(row):
            first = 0
            final = first
            found_first = False
            j = 0
            while j < col:
                if not found_first and mask[i, j] > 0:
                    found_first = True
                    first = j
                    final = col -1
                elif not found_first and mask[i, j] <= 0:
                    j += 1
                elif found_first and mask[i, final] <= 0:
                    final -= 1
                elif found_first and mask[i, final] > 0:
                    found_first = False
                    j = final + 1
                    mask[i, first:final+1] = 1

        
        # Reconstruct painting by columns
        for j in range(col):
            first = 0
            final = first
            found_first = False
            i = 0
            while i < row:
                if not found_first and mask[i, j] > 0:
                    found_first = True
                    first = i
                    final = row -1
                elif not found_first and mask[i, j] <= 0:
                    i += 1
                elif found_first and mask[final, j] <= 0:
                    final -= 1
                elif found_first and mask[final, j] > 0:
                    found_first = False
                    i = final + 1
                    mask[first:final+1, j] = 1

        
        ######
        # Erase small objects by columns
        RemoveBackground_2.__erase_small_obj_col(mask, threshold_erase_row)

        RemoveBackground_2.__erase_small_obj_row(mask, threshold_erase_col)

        RemoveBackground_2.__erase_small_obj_col(mask, threshold_erase_row)
        # Return the mask
        return mask.astype("uint8")

    @staticmethod
    def compute_removal_2(data):
        # Define thresholds
        threshold = 40
        threshold_2 = 30

        # Convert image to PIL image
        img = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(img)

        # Convert image to RGBA
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

        th = 10  # Window to compare peak
        # Search peak for red
        topR = 230
        for i in range(230, 0, -1):
            if l1[i] >= l1[topR]:
                topR = i
            elif topR - i > th:
                break

        # Search peak for green
        topG = 230
        for i in range(230, 0, -1):
            if l2[i] >= l2[topG]:
                topG = i
            elif topG - i > th:
                break

        # Search peak for blue
        topB = 230
        for i in range(230, 0, -1):
            if l3[i] >= l3[topB]:
                topB = i
            elif topB - i > th:
                break

        #########################
        # Search for first minimum after peak in red channel
        th = 10
        botR = topB
        for i in range(topB, 0, -1):
            if l1[i] <= l1[botR]:
                botR = i
            elif botR - i > th:
                break

        # Search for first minimum after peak in green channel
        botG = topG
        for i in range(topB, 0, -1):
            if l2[i] <= l2[botG]:
                botG = i
            elif botG - i > th:
                break

        # Search for first minimum after peak in blue channel
        botB = topB
        for i in range(topB, 0, -1):
            if l3[i] <= l3[botB]:
                botB = i
            elif botB - i > th:
                break

        # Define the values
        botR_shadow = botR - 120
        botG_shadow = botG - 120
        botB_shadow = botB - 120

        # Create the masks
        mask = np.logical_not((data[:, :, 0] > botR - threshold) & (data[:, :, 1] > botG - threshold) & (
                    data[:, :, 2] > botB - threshold))
        mask = np.logical_and(mask, np.logical_not(
            (data[:, :, 0] > botR_shadow - threshold_2 + 10) & (data[:, :, 0] < botR_shadow + threshold_2) & (
                        data[:, :, 1] > botG_shadow - threshold_2 + 10) & (
                        data[:, :, 1] < botG_shadow + threshold_2) & (
                        data[:, :, 2] > botB_shadow - threshold_2 + 10) & (data[:, :, 2] < botB_shadow + threshold_2)))

        # Return the mask
        return mask.astype("uint8")

    @staticmethod
    def __erase_small_obj_immersion(mask, ini, end):
        
        if end == ini:
            return ini
        elif end > ini:
            mid = ceil((end+ini)/2)
            if mask[mid] > 0:
                return RemoveBackground_2.__erase_small_obj_immersion(mask, mid, end)
            elif mask[mid] <= 0:
                return RemoveBackground_2.__erase_small_obj_immersion(mask, ini, mid-1)
        return 0  # this case should never happen
    
    @staticmethod
    def __erase_small_obj_col(mask, threshold):
        row, col = mask.shape
        for j in range(col):
            i = 0
            while i < row:
                if mask[i,j] > 0:
                    look = min(threshold, row - i - 1)
                    if mask[i+look,j] <= 0:
                        # search the other side
                        end = RemoveBackground_2.__erase_small_obj_immersion(mask[:,j], i, i + look - 1)
                        mask[i:end+1,j] = 0
                        i = end + 1
                    else:
                        i += look
                        i = RemoveBackground_2.__erase_small_obj_immersion(mask[:,j], i, row - 1)
                        i += 1
                else:
                    i += 1
    
    @staticmethod
    def __erase_small_obj_row(mask, threshold):
        row, col = mask.shape
        for i in range(row):
            j = 0
            while j < col:
                if mask[i,j] > 0:
                    look = min(threshold, col - j - 1)
                    if mask[i,j+look] <= 0:
                        # search the other side
                        end = RemoveBackground_2.__erase_small_obj_immersion(mask[i,:], j, j + look - 1)
                        mask[i,j:end+1] = 0
                        j = end + 1
                    else:
                        j += look
                        j = RemoveBackground_2.__erase_small_obj_immersion(mask[i,:], j, col - 1)
                        j += 1
                else:
                    j += 1





