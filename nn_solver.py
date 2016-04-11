import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.io import imread
from sklearn.preprocessing import LabelEncoder
from skimage.color import gray2rgb, rgb2gray
from skimage.color.adapt_rgb import adapt_rgb, each_channel
from skimage import filters
from skimage import exposure
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils


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
submit_name = 'benchmark_monte_carlo.csv'
debug = False
debug_n = 64
"""
Import images
"""
img_size = 30

# Train
path = "imgs"
train_folders = sorted(glob.glob(path + "/trainResized/*"))
train_names = []
for fol in train_folders:
    train_names += (glob.glob(fol + '/*'))

train_files = np.zeros((len(train_names), img_size, img_size, 3)).astype('float32')
train_labels = np.zeros((len(train_names),)).astype(str)
for i, name_file in enumerate(train_names):
    image = imp_img(name_file)
    train_files[i, :, :, :] = image
    train_labels[i] = name_file.split('/')[-2]

# Test
test_names = sorted(glob.glob(path + "/testResized/*"))
test_files = np.zeros((len(test_names), img_size, img_size, 3)).astype('float32')
for i, name_file in enumerate(test_names):
    image = imp_img(name_file)
    test_files[i, :, :, :] = image

train_files /= 255
test_files /= 255

label_encoder = LabelEncoder()
train_labels = label_encoder.fit_transform(train_labels)
print(train_files.shape, test_files.shape)
print(np.unique(train_labels))

"""
Image processing
"""
if debug:
    img_draw(train_files, train_names, debug_n)

# Contrast streching
for i, img_file in enumerate(train_files):
    train_files[i, :, :, :] = rescale_intensity_each(img_file)
for i, img_file in enumerate(test_files):
    test_files[i, :, :, :] = rescale_intensity_each(img_file)

if debug:
    img_draw(train_files, train_names, debug_n)

# Find borders
for i, img_file in enumerate(train_files):
    train_files[i, :, :, :] = sobel_each(img_file)
for i, img_file in enumerate(test_files):
    test_files[i, :, :, :] = sobel_each(img_file)

if debug:
    img_draw(train_files, train_names, debug_n)

train_files_gray = np.zeros((len(train_names), img_size, img_size)).astype('float32')
test_files_gray = np.zeros((len(test_names), img_size, img_size)).astype('float32')

# Change to gray
for i, img_file in enumerate(train_files):
    train_files_gray[i, :, :] = rgb2gray(img_file)
for i, img_file in enumerate(test_files):
    test_files_gray[i, :, :] = rgb2gray(img_file)

if debug:
    img_draw(train_files_gray, train_names, debug_n)

# Contrast streching
for i, img_file in enumerate(train_files_gray):
    p0, p100 = np.percentile(img_file, (0, 100))
    train_files_gray[i, :, :] = exposure.rescale_intensity(img_file, in_range=(p0, p100))
for i, img_file in enumerate(test_files_gray):
    p0, p100 = np.percentile(img_file, (0, 100))
    test_files_gray[i, :, :] = exposure.rescale_intensity(img_file, in_range=(p0, p100))

if debug:
    img_draw(train_files_gray, train_names, debug_n)

"""
Configure train/test by drivers and images per state
"""
driver_train_percent = 0.75
imgs_per_driver = 7
n_monte_carlo = 3

batch_size = 50
nb_classes = 10
nb_epoch = 20
# input image dimensions
img_rows, img_cols = img_size, img_size
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 3

drivers = pd.DataFrame.from_csv('driver_imgs_list.csv')
test_files_gray = test_files_gray.reshape(test_files_gray.shape[0], 1, img_rows, img_cols).astype('float32')

# convert class vectors to binary class matrices
train_labels_dummy = np_utils.to_categorical(train_labels, nb_classes)

test_results = []
for i_monte_carlo in range(n_monte_carlo):
    # Get driver - image relation table
    drivers_index = np.unique(drivers.index.values)

    np.random.seed(1000 * i_monte_carlo ** 2)
    cv_prob = np.random.sample(drivers_index.shape[0])
    train_cv_drivers = drivers_index[cv_prob < driver_train_percent]

    train_cv_ind = np.zeros((train_files_gray.shape[0],)).astype(bool)
    test_cv_ind = np.zeros((train_files_gray.shape[0],)).astype(bool)
    train_images = []
    # For each driver
    for driver in train_cv_drivers:
        driver_imgs = drivers.loc[train_cv_drivers]
        avail_states = np.unique(driver_imgs.classname.values)
        # For each state
        for state in avail_states:
            # Get imgs_per_driver images
            driver_state_imgs = driver_imgs.iloc[np.array(driver_imgs.classname == state)].img.values
            if imgs_per_driver < driver_state_imgs.shape[0]:
                train_img_index = np.random.choice(driver_state_imgs.shape[0], imgs_per_driver, replace=False)
                train_images += list(driver_state_imgs[train_img_index])
            else:
                train_images += list(driver_state_imgs)
    train_images = np.array(train_images)

    test_images = []
    test_cv_drivers = drivers_index[cv_prob >= driver_train_percent]
    for driver in test_cv_drivers:
        test_images += list(drivers.loc[driver].img.values)
    test_images = np.array(test_images)

    for i, file_name in enumerate(train_names):
        img_name = file_name.split('/')[-1]
        if img_name in train_images:
            train_cv_ind[i] = True
        if img_name in test_images:
            test_cv_ind[i] = True

    X_train, Y_train = train_files_gray[train_cv_ind, :, :].astype('float32'), train_labels_dummy[train_cv_ind, :]
    X_test, Y_test = train_files_gray[test_cv_ind, :, :].astype('float32'), train_labels_dummy[test_cv_ind, :]

    """
    Compile Model
    """
    # the data, shuffled and split between train and test sets
    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)

    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    np.random.seed(100 * i_monte_carlo ** 3)  # for reproducibility

    """
    CV model
    """
    model = Sequential()
    model.add(Convolution2D(nb_filters, nb_conv, nb_conv,
                            border_mode='valid', input_shape=(1, img_rows, img_cols)))
    model.add(Activation('relu'))

    """
    inner layers start
    """
    model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    model.add(Dropout(0.5))
    model.add(Activation('relu'))
    """
    inner layers stop
    """

    model.add(Flatten())
    model.add(Dense(50))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adadelta')
    model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
              show_accuracy=True, verbose=1, validation_data=(X_test, Y_test), shuffle=True)
    # score = model.evaluate(X_test, Y_test, show_accuracy=True, verbose=0)
    # print('Test score:', score[0])
    # print('Test accuracy:', score[1])

    """
    Get accuracy
    """
    # predicted_results = model.predict_classes(X_test, batch_size=batch_size, verbose=1)
    # print(label_encoder.inverse_transform(predicted_results))
    # print(label_encoder.inverse_transform(y_test))

    """
    Solve and submit test
    """
    predicted_results = model.predict_proba(test_files_gray, batch_size=batch_size, verbose=1)
    test_results.append(predicted_results)

sub_file = pd.DataFrame.from_csv('sample_submission.csv')
sub_file.iloc[:, :] = predicted_results
sub_file = sub_file.fillna(0.1)

predicted_results = np.zeros(sub_file.shape)
for mat in test_results:
    predicted_results += mat
predicted_results /= len(test_results)
print(predicted_results)

# Ordering sample index when needed
test_index = []
for file_name in test_names:
    test_index.append(file_name.split('/')[-1])
sub_file.index = test_index
sub_file.index.name = 'img'

sub_file.to_csv(submit_name)

# each_rescale_intensity -> each_border -> rgb2gray -> rescale_intensity:
