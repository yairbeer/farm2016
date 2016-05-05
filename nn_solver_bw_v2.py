import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss
from skimage.io import imread
from skimage.color import rgb2gray
import skimage.transform as tf
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.optimizers import SGD

"""
Functions
"""


def img_draw(im_arr, im_names, n_imgs):
    """
    Plot n_imgs images for debuging
    :param im_arr: image array
    :param im_names: image names list
    :param n_imgs: number of images
    :return: none
    """
    plt.figure(1)
    n_rows = int(np.sqrt(n_imgs))
    n_cols = n_imgs / n_rows
    for img_i in range(n_imgs):
        plt.subplot(n_cols, n_rows, img_i + 1)
        plt.title(im_names[img_i].split('/')[-1].split('.')[0])
        img = im_arr[img_i]
        plt.imshow(img, cmap='Greys_r')
    plt.show()


def img_rescale(img, scale):
    """
    rescale image
    :param img: image
    :param scale: scale factor
    :return: rescaled image
    """
    original_y, original_x = img.shape
    if scale > 1:
        img = tf.rescale(img, scale, clip=True)
        scaled_y, scaled_x = img.shape
        dx = (scaled_x - original_x) // 2
        dy = (scaled_y - original_y) // 2
        img = img[dy: (dy + original_y), dx: (dx + original_x)]
        return img
    else:
        tmp_img = np.zeros(img.shape)
        img = tf.rescale(img, scale)
        scaled_y, scaled_x = img.shape
        tmp_img[((original_y - scaled_y) // 2):((original_y - scaled_y) // 2 + scaled_y),
                ((original_x - scaled_x) // 2):((original_x - scaled_x) // 2 + scaled_x)] = img
        return tmp_img


def img_updown(img, up):
    """
    Translate image up or down
    :param img: image
    :param up: translate up factor
    :return: translated image
    """
    h = img.shape[0]
    up_pixels = int(h * up)
    tmp_img = np.zeros(img.shape)
    if up_pixels > 0:
        tmp_img[up_pixels:, :] = img[: - up_pixels, :]
    else:
        if up_pixels < 0:
            tmp_img[: up_pixels, :] = img[-up_pixels:, :]
        else:
            tmp_img = img
    return tmp_img


def img_leftright(img, right):
    """
    Translate image left or right
    :param img: image
    :param right: translate right factor
    :return: translated image
    """
    w = img.shape[1]
    right_pixels = int(w * right)
    tmp_img = np.zeros(img.shape)
    if right_pixels > 0:
        tmp_img[:, right_pixels:] = img[:, : (-1 * right_pixels)]
    else:
        if right_pixels < 0:
            tmp_img[:, : right_pixels] = img[:, (-1 * right_pixels):]
        else:
            tmp_img = img
    return tmp_img


def imp_batch(img_names):
    """
    Read and preprocess batch of images
    :param img_names: array of image file names
    :return: image array
    """
    img_arr = np.zeros((img_names.shape[0], img_rows, img_cols))
    for i, img in enumerate(img_names):
        # read
        img = imread(img)
        # convert to gray
        img = rgb2gray(img)
        img_arr[i] = img
    img_reshaped = np.zeros((img_names.shape[0], 1, img_rows, img_cols)).astype('float32')
    img_reshaped[:, 0, :, :] = img_arr
    return img_reshaped


def eval_randomized(data_names, data_results, n_test, nn_model):
    """
    Evaluate score only on part of the data
    :param data_names: images paths
    :param data_results: Corresponding results
    :param n_test: number of evaluated images
    :param nn_model: CNN model
    :return: none
    """
    image_indexes = np.random.choice(range(data_names.shape[0]), n_test, replace=False)
    data_names = data_names[image_indexes]
    data_results = data_results[image_indexes, :]
    img_arr = np.zeros((n_test, 1, img_rows, img_cols)).astype('float32')
    for i, img in enumerate(data_names):
        # read
        img = imread(img)
        # convert to gray
        img = rgb2gray(img)
        img_arr[i, 0] = img
    score = nn_model.evaluate(img_arr, data_results, verbose=0, show_accuracy=True)
    return score


def predict_batch(data_names, classes, nn_model):
    """
    predict probability in batches
    :param data_names: images paths
    :param nn_model: CNN model
    :return: none
    """
    n_test = data_names.shape[0]
    predicted_prob = np.zeros((n_test, classes))

    print('calculating in %d batches' % len(range(0, X_train_cp.shape[0], batch_size)))
    for batch_i in range(0, n_test, batch_size):
        print('batch %d' % batch_i)
        # Update learning rate if needed
        if (batch_i + batch_size) < n_test:
            batch_names = data_names[batch_i: batch_i + batch_size]
            batch_train = imp_batch(batch_names)
            predicted_prob[batch_i: batch_i + batch_size, :] = nn_model.predict_proba(batch_train,
                                                                                      batch_size=batch_size, verbose=1)
        else:
            batch_names = data_names[batch_i:]
            batch_train = imp_batch(batch_names)
            predicted_prob[batch_i:, :] = nn_model.predict_proba(batch_train, batch_size=batch_size, verbose=1)
    return predicted_prob


def cnn_model():
    """
    Create CNN model
    :return: model
    """
    model = Sequential()
    model.add(Convolution2D(64, nb_conv, nb_conv,
                            border_mode='valid', input_shape=(1, img_rows, img_cols)))
    model.add(Activation('relu'))
    """
    inner layers start
    """
    model.add(Convolution2D(64, nb_conv, nb_conv))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    model.add(Dropout(0.5))

    model.add(Convolution2D(64, nb_conv, nb_conv))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, nb_conv, nb_conv))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    model.add(Dropout(0.5))

    model.add(Convolution2D(64, nb_conv, nb_conv))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, nb_conv, nb_conv))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    """
    inner layers stop
    """

    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    sgd = SGD(lr=0.03, decay=1e-5, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd)
    return model

"""
Vars
"""
# Output file name
submit_name = 'bw_160x120_v2.csv'

# Train and test location
train_sub = "/trainResized160/"
test_sub = "/testResized160/"

# Input image size
img_size_y = 120
img_size_x = 160

# To debug?
debug = False
# How many images to show?
debug_n = 100

# Number of folds per training set, 0 means no CV
n_fold = 2

# Number of ensembles of drivers
n_ensemble = 1
# What percent of the drivers to use in each ensemble
percent_drivers = 1.0

# input image dimensions
img_rows, img_cols = img_size_y, img_size_x
# NN's batch size
batch_size = 64
# Number of training batches
nb_batch = 3000
# At what frequency of batches to print prediction results
man_verbose = 30
# Number of NN epochs
nb_epoch = 100
# Output classes
nb_classes = 10
# Number of images for random evaluating
n_eval = 2000

# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 3
# learning rate update, index is the batch round
lr_updates = {0: 0.01, 1001: 0.003}

"""
Start program
"""

# Read images
# Train
path = "imgs"
train_folders = sorted(glob.glob(path + train_sub + "*"))
train_names = []
for fol in train_folders:
    train_names += (glob.glob(fol + '/*'))
train_names = np.array(train_names)

train_labels = np.zeros((len(train_names),)).astype(str)
for i, name_file in enumerate(train_names):
    train_labels[i] = name_file.split('/')[-2]

# Test
test_names = sorted(glob.glob(path + test_sub + "*"))
test_names = np.array(test_names)

label_encoder = LabelEncoder()
train_labels = label_encoder.fit_transform(train_labels)

"""
Configure train/test by drivers and images per state
"""
# Read relation table of drivers and
drivers = pd.DataFrame.from_csv('driver_imgs_list.csv')
# Get all the drivers
drivers_index = np.unique(drivers.index.values)

# convert class vectors to binary class matrices
train_labels_dummy = np_utils.to_categorical(train_labels, nb_classes)

if n_fold:
    test_results = []
    test_acc = []
    for i_fold in range(n_fold):
        print('Fold %d' % i_fold)

        # Seed for repeatability
        np.random.seed(1000 * i_fold)
        train_test_driver_index = np.random.choice(range(drivers_index.shape[0]), drivers_index.shape[0],
                                                   replace=False)
        train_driver_index = train_test_driver_index[: int(drivers_index.shape[0] * (1 - 1/n_fold))]
        test_driver_index = train_test_driver_index[int(drivers_index.shape[0] * (1 - 1/n_fold)):]

        # The number of drivers is cv_prob percent of the data
        train_cv_drivers = []
        for i_train in range(n_ensemble):
            train_cv_drivers.append(np.random.choice(drivers_index[train_driver_index],
                                                     int(train_driver_index.shape[0] * percent_drivers),
                                                     replace=False))
        train_cv_ind = np.zeros((train_names.shape[0], n_ensemble)).astype(bool)
        test_cv_ind = np.zeros(train_names.shape).astype(bool)

        train_images = []
        # For each driver
        for i_train in range(n_ensemble):
            train_images.append([])
            for driver in train_cv_drivers[i_train]:
                driver_imgs = drivers.loc[driver]
                avail_states = np.unique(driver_imgs.classname.values)
                # For each driving state
                for state in avail_states:
                    # Get imgs_per_driver images (using all the images can overfit)
                    driver_state_imgs = driver_imgs.iloc[np.array(driver_imgs.classname == state)].img.values
                    driver_state_imgs = map(lambda x: str('imgs' + train_sub + state + '/' + x), driver_state_imgs)
                    train_images[i_train] += list(driver_state_imgs)
            train_images[i_train] = np.array(train_images[i_train])

        test_images = []
        # Use all images of the test driver as test
        test_cv_drivers = drivers_index[test_driver_index]
        for driver in test_cv_drivers:
            driver_imgs = drivers.loc[driver]
            avail_states = np.unique(driver_imgs.classname.values)
            for state in avail_states:
                driver_state_imgs = driver_imgs.iloc[np.array(driver_imgs.classname == state)].img.values
                driver_state_imgs = map(lambda x: str('imgs' + train_sub + state + '/' + x), driver_state_imgs)
                test_images += list(driver_state_imgs)
        test_images = np.array(test_images)

        for i, file_name in enumerate(train_names):
            for i_train in range(n_ensemble):
                if file_name in train_images[i_train]:
                    train_cv_ind[i, i_train] = True
            if file_name in test_images:
                test_cv_ind[i] = True
                # Get the train / test split

        X_train = []
        Y_train = []
        X_train_n_imgs = []
        for i_train in range(n_ensemble):
            X_train.append(train_names[train_cv_ind[:, i_train]])
            Y_train.append(train_labels_dummy[train_cv_ind[:, i_train], :])
        X_test, Y_test = train_names[test_cv_ind], train_labels_dummy[test_cv_ind, :]

        """
        Compile Model
        """
        for i_train in range(n_ensemble):
            print("Train set %d has %d samples" % (i_train, X_train[i_train].shape[0]))
        print(X_test.shape[0], 'test samples')

        np.random.seed(1000 * i_fold + 10)  # for reproducibility

        """
        CV model
        """
        # Get image preprocessing values
        cv_predict_test = []
        for i_train in range(n_ensemble):
            print('Ensemble trainer %d' % i_train)
            # Build model
            train_model = cnn_model()
            batch_count = 0
            for epoch_i in range(nb_epoch):
                print('Epoch %d' % epoch_i)
                # For each training set copy training set
                X_train_cp = np.array(X_train[i_train], copy=True)
                np.random.seed(epoch_i)
                # Randomize batch order
                batch_order = np.random.choice(range(X_train_cp.shape[0]), X_train_cp.shape[0],
                                               replace=False)
                X_train_cp = X_train_cp[batch_order]
                Y_train_cp = Y_train[i_train][batch_order]
                # Solve epoch
                for batch_i in range(0, X_train_cp.shape[0], batch_size):
                    print('batch %d' % batch_count)
                    # Update learning rate if needed
                    if batch_count in lr_updates:
                        print('lr changed to %f' % lr_updates[batch_count])
                        train_model.optimizer.lr.set_value(lr_updates[batch_count])
                    if (batch_i + batch_size) < X_train_cp.shape[0]:
                        batch_names = X_train_cp[batch_i: batch_i + batch_size]
                        batch_train = imp_batch(batch_names)
                        train_model.train_on_batch(batch_train, Y_train_cp[batch_i: batch_i + batch_size],
                                                   accuracy=True)
                    else:
                        batch_names = X_train_cp[batch_i:]
                        batch_train = imp_batch(batch_names)
                        train_model.train_on_batch(batch_train, Y_train_cp[batch_i:], accuracy=True)
                    batch_count += 1
                    # Stop training current batch if gotten to nb_batches
                    if man_verbose:
                        if not (batch_count % man_verbose):
                            print('For batch %d' % batch_count)
                            score = eval_randomized(X_train_cp, Y_train_cp, n_eval, train_model)
                            print('Train score: %.2f, train accuracy: %.3f' % (score[0], score[1]))
                            score = eval_randomized(X_test, Y_test, n_eval, train_model)
                            print('Test score: %.2f, test accuracy: %.3f' % (score[0], score[1]))
                    if batch_count == nb_batch:
                        break
                # Stop training current batch if gotten to nb_batches
                if batch_count == nb_batch:
                    break

            # Fit calculated model to the test data
            cv_predict_test.append(predict_batch(train_labels[test_cv_ind], 10, train_model))

        cv_ensemble_predicted_results = np.zeros(cv_predict_test[0].shape)
        for mat in cv_predict_test:
            cv_ensemble_predicted_results += mat
            cv_ensemble_predicted_results /= n_ensemble
        print('The average test score %.3f' % log_loss(train_labels[test_cv_ind], cv_ensemble_predicted_results))

"""
Solve and submit test
"""
train_cv_drivers = []
for i_train in range(n_ensemble):
    train_cv_drivers.append(np.random.choice(drivers_index,
                                             int(drivers_index.shape[0] * percent_drivers), replace=False))

train_images = []
# For each driver
for i_train in range(n_ensemble):
    train_images.append([])
    for driver in train_cv_drivers[i_train]:
        driver_imgs = drivers.loc[driver]
        avail_states = np.unique(driver_imgs.classname.values)
        # For each driving state
        for state in avail_states:
            # Get imgs_per_driver images (using all the images can overfit)
            driver_state_imgs = driver_imgs.iloc[np.array(driver_imgs.classname == state)].img.values
            driver_state_imgs = map(lambda x: str('imgs' + train_sub + state + '/' + x), driver_state_imgs)
            train_images[i_train] += list(driver_state_imgs)
    train_images[i_train] = np.array(train_images[i_train])

train_ind = np.zeros((train_names.shape[0], n_ensemble)).astype(bool)
for i, file_name in enumerate(train_names):
    for i_train in range(n_ensemble):
        if file_name in train_images[i_train]:
            train_ind[i, i_train] = True

X_train = []
Y_train = []
for i_train in range(n_ensemble):
    X_train.append(train_names[train_ind[:, i_train]])
    Y_train.append(train_labels_dummy[train_ind[:, i_train], :])

# Get image preprocessing values
predict_test = []
for i_train in range(n_ensemble):
    print('Ensemble trainer %d' % i_train)
    # Build model
    train_model = cnn_model()
    batch_count = 0
    for epoch_i in range(nb_epoch):
        print('Epoch %d' % epoch_i)
        # For each training set copy training set
        X_train_cp = np.array(X_train[i_train], copy=True)
        np.random.seed(epoch_i)
        # Randomize batch order
        batch_order = np.random.choice(range(X_train_cp.shape[0]), X_train_cp.shape[0],
                                       replace=False)
        X_train_cp = X_train_cp[batch_order]
        Y_train_cp = Y_train[i_train][batch_order]
        # Solve epoch
        for batch_i in range(0, X_train_cp.shape[0], batch_size):
            print('batch %d' % batch_count)
            # Update learning rate if needed
            if batch_count in lr_updates:
                print('lr changed to %f' % lr_updates[batch_count])
                train_model.optimizer.lr.set_value(lr_updates[batch_count])
            if (batch_i + batch_size) < X_train_cp.shape[0]:
                batch_names = X_train_cp[batch_i: batch_i + batch_size]
                batch_train = imp_batch(batch_names)
                train_model.train_on_batch(batch_train, Y_train_cp[batch_i: batch_i + batch_size],
                                           accuracy=True)
            else:
                batch_names = X_train_cp[batch_i:]
                batch_train = imp_batch(batch_names)
                train_model.train_on_batch(batch_train, Y_train_cp[batch_i:], accuracy=True)
            batch_count += 1
            # Stop training current batch if gotten to nb_batches
            if man_verbose:
                if not (batch_count % man_verbose):
                    print('For batch %d' % batch_count)
                    score = eval_randomized(X_train_cp, Y_train_cp, n_eval, train_model)
                    print('Train score: %.2f, train accuracy: %.3f' % (score[0], score[1]))
            if batch_count == nb_batch:
                break
        # Stop training current batch if gotten to nb_batches
        if batch_count == nb_batch:
            break

    # Fit calculated model to the test data
    predict_test.append(predict_batch(test_names, 10, train_model))
    print(predict_test[0])

ensemble_predicted_results = np.zeros(predict_test[0].shape)
for mat in predict_test:
    ensemble_predicted_results += mat
    ensemble_predicted_results /= n_ensemble

sub_file = pd.DataFrame.from_csv('sample_submission.csv')

print(ensemble_predicted_results)

sub_file.iloc[:, :] = ensemble_predicted_results
sub_file = sub_file.fillna(0.1)

# Ordering sample index when needed
test_index = []
for file_name in test_names:
    test_index.append(file_name.split('/')[-1])
sub_file.index = test_index
sub_file.index.name = 'img'

sub_file.to_csv(submit_name)
