import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy import fftpack

def dct_2d(img):
    return fftpack.dct(
        fftpack.dct(img, axis=0, norm='ortho'), axis=1, norm='ortho')

def idct_2d(img):
    return fftpack.idct(
        fftpack.idct(img, axis=0, norm='ortho'), axis=1, norm='ortho')

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def dct_image(im):
    dct = np.zeros(im.shape)
    for i in range(0,im.shape[0],8):
        for j in range(0, im.shape[1],8):
            dct[i:(i+8),j:(j+8)] = dct_2d(im[i:(i+8),j:(j+8)])
    return dct

def idct_image(im):
    idct = np.zeros(im.shape)
    for i in range(0,im.shape[0],8):
        for j in range(0, im.shape[1],8):
            idct[i:(i+8),j:(j+8)] = idct_2d(im[i:(i+8),j:(j+8)])
    return idct

im = rgb2gray(mpimg.imread('test.jpg'))
dct = dct_image(im)

f, axarr = plt.subplots(2,2)
f.suptitle('DCT Demonstration')
axarr[0, 0].imshow(im, cmap='gray')
axarr[0, 0].set_title('Original Image')

axarr[1, 0].imshow(dct,cmap='gray',vmax=np.max(dct)*0.01,vmin = 0)
axarr[1, 0].set_title('DCT')

# Threshold DCT
thresh = 0.012
dct_thresh = dct * (abs(dct) > (thresh*np.max(dct)))

percent_kept = np.sum(dct_thresh != 0.0) / (im.shape[0]*im.shape[1]*1.0)

axarr[1, 1].imshow(dct_thresh, cmap='gray', vmax=np.max(dct)*0.01,vmin = 0)
axarr[1, 1].set_title('DCT with %f%% of coefficients removed' % \
                 ((1-percent_kept)*100.0))

axarr[0, 1].imshow(idct_image(dct_thresh), cmap='gray')
axarr[0, 1].set_title('Compressed Image')

plt.show()
