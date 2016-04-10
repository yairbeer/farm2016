import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.color import gray2rgb, rgb2gray
from skimage.color.adapt_rgb import adapt_rgb, each_channel
from skimage import filters
from skimage import exposure


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

# Train
path = "imgs"
train_folders = sorted(glob.glob(path + "/train/*"))
train_names = []
for fol in train_folders:
    train_names += (glob.glob(fol + '/*'))

train_files = np.zeros((len(train_names), 640, 480, 3)).astype('float32')
train_labels = np.zeros((len(train_names),)).astype(str)
for i, name_file in enumerate(train_names):
    image = imp_img(name_file)
    train_files[i, :, :, :] = image
    train_labels[i] = name_file.split('/')[-2]

train_files /= 255

print(train_files.shape)

"""
Image processing
"""
if debug:
    img_draw(train_files, train_names, debug_n)

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
