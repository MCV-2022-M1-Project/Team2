import argparse
import collections
import pickle
import cv2
import numpy as np
from imutils.paths import list_images
from skimage.restoration import (denoise_wavelet, estimate_sigma)
import math
class RemoveNoise:
    #check if image has other type of noise TODO

    qsd1_w3_y = [0,1,6,8,11,13,16,17,20,23,24,26,29] # noisy images from qsd1_w3 dataset
    qsd1_w3_n = [2,3,4,5,7,9,10,12,14,15,18,19,21,22,25,27,28] # not noisy images from qsd1_w3 dataset

    def __init__(self,image):
        self.image = image

    def hasSaltNPepper(self, verbose = True):
        #check if image has salt and pepper noise
        m_filtered = cv2.medianBlur(self.image, 3)
        noise = cv2.subtract(m_filtered, self.image)
        #count the noise pixels and check how many pixels there are relative to image size
        n_pix = np.sum(noise > 30)/3
        if n_pix>((self.image.shape[0]*self.image.shape[1])/220):
            if verbose:
                print("Salt and pepper noise detected!")
            return True
        else:
            return False

    def denoise_image(self):
        #denoise image depending on type of noise

        if self.hasSaltNPepper():
            #m_filtered = cv2.medianBlur(self.image, 3)
            #image_gray = cv2.cvtColor(m_filtered, cv2.COLOR_BGR2GRAY)
            #edges = np.absolute(cv2.Sobel(image_gray,cv2.CV_16S,1,1))
            #edges = ((edges-edges.min())/(edges.max()-edges.min())*255).astype("uint8")
            sigma_est = estimate_sigma(self.image, channel_axis=-1, average_sigmas=True)
            im_visushrink2 = denoise_wavelet(self.image, channel_axis=-1, convert2ycbcr=True,
                                            method='VisuShrink', mode='soft',
                                            sigma=sigma_est/2, rescale_sigma=True)
            im_visushrink2 = (im_visushrink2 * 255).astype("uint8")
            #im_visushrink2[:,:,0] = cv2.add(im_visushrink2[:,:,0], edges)
            #im_visushrink2[:,:,1] = cv2.add(im_visushrink2[:,:,1], edges)
            #im_visushrink2[:,:,2] = cv2.add(im_visushrink2[:,:,2], edges)
            return im_visushrink2
        else:
            #m_filtered = cv2.medianBlur(self.image, 3)
            #image_gray = cv2.cvtColor(m_filtered, cv2.COLOR_BGR2GRAY)
            #edges = np.absolute(cv2.Sobel(image_gray,cv2.CV_16S,1,1))
            #edges = ((edges-edges.min())/(edges.max()-edges.min())*255).astype("uint8")
            sigma_est = estimate_sigma(self.image, channel_axis=-1, average_sigmas=True)
            im_visushrink2 = denoise_wavelet(self.image, channel_axis=-1, convert2ycbcr=True,
                                            method='VisuShrink', mode='soft',
                                            sigma=sigma_est/4, rescale_sigma=True)
            im_visushrink2 = (im_visushrink2 * 255).astype("uint8")
            #im_visushrink2[:,:,0] = cv2.add(im_visushrink2[:,:,0], edges)
            #im_visushrink2[:,:,1] = cv2.add(im_visushrink2[:,:,1], edges)
            #im_visushrink2[:,:,2] = cv2.add(im_visushrink2[:,:,2], edges)
            return im_visushrink2

    def noise_accuracy_qsd1_w3(self, name):
        #getting the name of the image tells if the detection is correct
        gt_noisy = int(name) in self.qsd1_w3_y

        return gt_noisy == self.hasSaltNPepper(verbose=True)

