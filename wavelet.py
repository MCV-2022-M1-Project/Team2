# Import required packages
import cv2
from skimage.restoration import (denoise_wavelet, estimate_sigma)

name = '../dataset/qsd2_w3/00010.jpg'
imagecv = cv2.imread(name)


#sigma_est = estimate_sigma(imagecv, channel_axis=-1, average_sigmas=True)
#im_visushrink2 = denoise_wavelet(imagecv, channel_axis=-1, convert2ycbcr=True,
#                                 method='VisuShrink', mode='soft',
#                                 sigma=sigma_est/4, rescale_sigma=True)
im = denoise_wavelet(imagecv, channel_axis=-1, convert2ycbcr=True,
                                            method='BayesShrink', mode='hard', rescale_sigma=True)

cv2.imwrite("denoised.png", im*255)