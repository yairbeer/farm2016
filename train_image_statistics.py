import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.io import imread, imsave
from skimage.color import gray2rgb, rgb2gray
from skimage.color.adapt_rgb import adapt_rgb, each_channel
from skimage import filters
from skimage import exposure
from skimage.transform import resize
import os


def img_draw(im_arr, im_names, n_imgs):
    plt.figure(1)
    n_rows = int(np.sqrt(n_imgs))
    n_cols = n_imgs / n_rows
    for img_i in range(n_imgs):
        plt.subplot(n_cols, n_rows, img_i + 1)
        plt.title(im_names[img_i].split('/')[-1].split('.')[0])
        if len(im_arr.shape) == 4:
            img = im_arr[img_i]
        else:
            img = im_arr[img_i]
        plt.imshow(img)
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
Vars
"""
submit_name = 'benchmark.csv'
debug = False
debug_n = 64
"""
Import images
"""
# Set path of data files
path = "imgs"

if not os.path.exists(path + "/trainResized"):
    os.makedirs(path + "/trainResized")


img_size = 40

train_files_unlabeled = sorted(glob.glob(path + "/trainResized/*"))

train_files_labeled = []
for fol in train_files_unlabeled:
    train_files_labeled.append(glob.glob(fol + '/*'))

avg_classes = []
for fol in train_files_labeled:
    for im_name in fol:
        image = imread(im_name)

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
