import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.io import imread, imsave
from skimage.color import gray2rgb, rgb2gray
from skimage.color.adapt_rgb import adapt_rgb, each_channel
from skimage import filters
from skimage import exposure
import os


def img_draw(im_arr):
    plt.figure(1)
    n_rows = int(4)
    n_cols = 3
    for img_i in range(10):
        plt.subplot(n_cols, n_rows, img_i + 1)
        plt.title('C' + str(img_i))
        plt.imshow(im_arr[img_i])
    plt.show()


def imp_img(img_name):
    # read
    img = imread(img_name)
    # if gray convert to color
    if len(img.shape) == 2:
        img = gray2rgb(img)
    return img


@adapt_rgb(each_channel)
def sobel_each(image):
    return filters.sobel(image)


@adapt_rgb(each_channel)
def rescale_intensity_each(image):
    plow, phigh = np.percentile(image, (0, 100))
    return np.clip(exposure.rescale_intensity(image, in_range=(plow, phigh)), 0, 1)

"""
Import images
"""
# Set path of data files
path = "imgs"

img_size_y = 48
img_size_x = 64

train_files_unlabeled = sorted(glob.glob(path + "/trainResized/*"))

train_files_labeled = []
for fol in train_files_unlabeled:
    train_files_labeled.append(glob.glob(fol + '/*'))

avg_imgs = []
for fol in train_files_labeled:
    average_im = np.zeros((img_size_y, img_size_x, 3))
    for im_name in fol:
        image = imread(im_name)
        image = image / 256
        image = rescale_intensity_each(image)
        average_im += image
    average_im /= len(fol)
    average_im = rescale_intensity_each(average_im)
    avg_imgs.append(average_im)

img_draw(avg_imgs)

avg_imgs = []
for fol in train_files_labeled:
    average_im = np.zeros((img_size_y, img_size_x, 3))
    for im_name in fol:
        image = imread(im_name)
        image = image / 256
        image = rescale_intensity_each(image)
        image = sobel_each(image)
        average_im += image
    average_im /= len(fol)
    average_im = rescale_intensity_each(average_im)
    avg_imgs.append(average_im)

img_draw(avg_imgs)

"""
Image processing
"""

# # Contrast streching
# for i, img_file in enumerate(train_files):
#     train_files[i, :, :, :] = rescale_intensity_each(img_file)
#
# if debug:
#     img_draw(train_files, train_names, debug_n)
#
# # Find borders
# for i, img_file in enumerate(train_files):
#     train_files[i, :, :, :] = sobel_each(img_file)
#
# if debug:
#     img_draw(train_files, train_names, debug_n)
#
# train_files_gray = np.zeros((len(train_names), img_size, img_size)).astype('float32')
#
# # Change to gray
# for i, img_file in enumerate(train_files):
#     train_files_gray[i, :, :] = rgb2gray(img_file)
#
# if debug:
#     img_draw(train_files_gray, train_names, debug_n)
#
# # Contrast streching
# for i, img_file in enumerate(train_files_gray):
#     p0, p100 = np.percentile(img_file, (0, 100))
#     train_files_gray[i, :, :] = exposure.rescale_intensity(img_file, in_range=(p0, p100))
#
# if debug:
#     img_draw(train_files_gray, train_names, debug_n)
