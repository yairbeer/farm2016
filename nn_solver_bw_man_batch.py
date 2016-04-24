import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.color.adapt_rgb import adapt_rgb, each_channel
from skimage import filters
from skimage import exposure
from skimage import feature
import skimage.transform as tf
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.optimizers import SGD


def img_draw(im_arr, im_names, n_imgs):
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


def img_rotate(img, rotate, corner_deg_chance):
    rot_chance = np.random.random()
    if rot_chance < corner_deg_chance:
        return tf.rotate(img, 90)
    if corner_deg_chance <= rot_chance < (corner_deg_chance * 2):
        return tf.rotate(img, 180)
    if (corner_deg_chance * 2) <= rot_chance < (corner_deg_chance * 3):
        return tf.rotate(img, 270)
    return tf.rotate(img, rotate)


def imp_img(img_name):
    # read
    img = imread(img_name)
    # convert to gray
    img = rgb2gray(img)
    return img


def cnn_model():
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

    """
    inner layers stop
    """
    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    sgd = SGD(lr=0.03, decay=1e-5, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd)
    return model


@adapt_rgb(each_channel)
def corner_each(image):
    return feature.corner_harris(image)


@adapt_rgb(each_channel)
def sobel_each(image):
    return filters.sobel(image)


@adapt_rgb(each_channel)
def rescale_intensity_each(image, low, high):
    plow, phigh = np.percentile(image, (low, high))
    return np.clip(exposure.rescale_intensity(image, in_range=(plow, phigh)), 0, 1)

"""
Vars
"""
submit_name = 'rgb_32x24_man_subsample.csv'
debug = False
debug_n = 100
"""
Import images
"""
img_size_y = 24
img_size_x = 32

# Train
path = "imgs"
train_folders = sorted(glob.glob(path + "/trainResizedSmall/*"))
train_names = []
for fol in train_folders:
    train_names += (glob.glob(fol + '/*'))

train_files = np.zeros((len(train_names), img_size_y, img_size_x)).astype('float32')
train_labels = np.zeros((len(train_names),)).astype(str)
for i, name_file in enumerate(train_names):
    image = imp_img(name_file)
    train_files[i, :, :] = image
    train_labels[i] = name_file.split('/')[-2]

# Test
test_names = sorted(glob.glob(path + "/testResizedSmall/*"))
test_files = np.zeros((len(test_names), img_size_y, img_size_x)).astype('float32')
for i, name_file in enumerate(test_names):
    image = imp_img(name_file)
    test_files[i, :, :] = image

train_files /= 255
test_files /= 255

label_encoder = LabelEncoder()
train_labels = label_encoder.fit_transform(train_labels)
# print(train_files.shape, test_files.shape)
# print(np.unique(train_labels))

"""
Image processing
"""
if debug:
    img_draw(train_files, train_names, debug_n)

"""
Configure train/test by drivers and images per state
"""

n_montecarlo = 1
n_fold = 5
n_ensemble = 3
percent_drivers = 0.75
imgs_per_driver = 10000

batch_size = 256
nb_classes = 10
nb_epoch = 20
# input image dimensions
img_rows, img_cols = img_size_y, img_size_x
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 3
# lr update
lr_updates = {0: 0.03}

drivers = pd.DataFrame.from_csv('driver_imgs_list.csv')
train_files_cnn = np.zeros((train_files.shape[0], 1, img_rows, img_cols)).astype('float32')
test_files_cnn = np.zeros((test_files.shape[0], 1, img_rows, img_cols)).astype('float32')

# convert class vectors to binary class matrices
train_labels_dummy = np_utils.to_categorical(train_labels, nb_classes)

for i_mc in range(n_montecarlo):
    test_results = []
    test_acc = []
    for i_fold in range(n_fold):
        # Get all the drivers
        drivers_index = np.unique(drivers.index.values)

        # Seed for repeatability
        np.random.seed(1000 * i_fold + 100 * i_mc)
        train_test_driver_index = np.random.choice(range(drivers_index.shape[0]), drivers_index.shape[0], replace=False)
        train_driver_index = train_test_driver_index[: int(drivers_index.shape[0] * (1 - 1/n_fold))]
        test_driver_index = train_test_driver_index[int(drivers_index.shape[0] * (1 - 1/n_fold)):]

        # On Average the number of drivers is cv_prob percent of the data
        train_cv_drivers = []
        for i_train in range(n_ensemble):
            train_cv_drivers.append(np.random.choice(drivers_index[train_driver_index],
                                                     int(train_driver_index.shape[0] * percent_drivers), replace=False))
        train_cv_ind = np.zeros((train_files.shape[0], n_ensemble)).astype(bool)
        test_cv_ind = np.zeros((train_files.shape[0],)).astype(bool)

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
                    if imgs_per_driver < driver_state_imgs.shape[0]:
                        train_img_index = np.random.choice(driver_state_imgs.shape[0], imgs_per_driver, replace=False)
                        train_images[i_train] += list(driver_state_imgs[train_img_index])
                    else:
                        train_images[i_train] += list(driver_state_imgs)
            train_images[i_train] = np.array(train_images[i_train])

        test_images = []
        # Use all images of the test driver as test
        test_cv_drivers = drivers_index[test_driver_index]
        for driver in test_cv_drivers:
            test_images += list(drivers.loc[driver].img.values)
        test_images = np.array(test_images)

        for i, file_name in enumerate(train_names):
            img_name = file_name.split('/')[-1]
            for i_train in range(n_ensemble):
                if img_name in train_images[i_train]:
                    train_cv_ind[i, i_train] = True
            if img_name in test_images:
                test_cv_ind[i] = True

        # Get the train / test split
        X_train = []
        Y_train = []
        X_train_n_imgs = []
        for i_train in range(n_ensemble):
            X_train.append(train_files_cnn[train_cv_ind[:, i_train]].astype('float32'))
            Y_train.append(train_labels_dummy[train_cv_ind[:, i_train], :])
        X_test, Y_test = train_files_cnn[test_cv_ind].astype('float32'), train_labels_dummy[test_cv_ind, :]

        """
        Compile Model
        """
        for i_train in range(n_ensemble):
            print("Train set %d has %d samples" % (i_train, X_train[i_train].shape[0]))
        print(X_test.shape[0], 'test samples')

        np.random.seed(1000 * i_fold + 100 * i_mc + 10)  # for reproducibility

        """
        CV model
        """
        # Train cnn models
        train_models = []
        for i_train in range(n_ensemble):
            train_models.append(cnn_model())

        # For each epoch
        for epoch_i in range(nb_epoch):
            print('Epoch %d' % epoch_i)
            # Get image preprocessing values
            X_train_cp = []
            rotate_angle = []
            rescale_fac = []
            right_move = []
            up_move = []
            shear = []
            afine_tf = []
            # For each training set
            for i_train in range(n_ensemble):
                np.random.seed(epoch_i)
                X_train_cp.append(np.array(X_train[i_train], copy=True))
                rotate_angle.append(np.random.normal(0, 3, X_train_cp[i_train].shape[0]))
                rescale_fac.append(np.random.normal(1, 0.05, X_train_cp[i_train].shape[0]))
                right_move.append(np.random.normal(0, 0.05, X_train_cp[i_train].shape[0]))
                up_move.append(np.random.normal(0, 0.05, X_train_cp[i_train].shape[0]))
                shear.append(np.random.normal(0, 3, X_train_cp[i_train].shape[0]))
                shear[i_train] = np.deg2rad(shear[i_train])
            # For each training set copy training set
            for i_train in range(n_ensemble):
                # Update learning rate if needed
                if epoch_i in lr_updates:
                    print('lr changed to %f' % lr_updates[epoch_i])
                    train_models[i_train].optimizer.lr.set_value(lr_updates[epoch_i])
                # Preprocess images
                for img_i in range(X_train_cp[i_train].shape[0]):
                    afine_tf = tf.AffineTransform(shear=shear[i_train][img_i])
                    X_train_cp[i_train][img_i, 0] = tf.warp(X_train_cp[i_train][img_i, 0], afine_tf)
                    X_train_cp[i_train][img_i, 0] = img_rotate(X_train_cp[i_train][img_i, 0],
                                                               rotate_angle[i_train][img_i], -1)
                    X_train_cp[i_train][img_i, 0] = img_rescale(X_train_cp[i_train][img_i, 0],
                                                                rescale_fac[i_train][img_i])
                    X_train_cp[i_train][img_i, 0] = img_leftright(X_train_cp[i_train][img_i, 0],
                                                                  right_move[i_train][img_i])
                    X_train_cp[i_train][img_i, 0] = img_updown(X_train_cp[i_train][img_i, 0],
                                                               up_move[i_train][img_i])
                # Randomize batch order
                batch_order = np.random.choice(range(X_train_cp[i_train].shape[0]), X_train_cp[i_train].shape[0],
                                               replace=False)
                X_train_cp[i_train] = X_train_cp[i_train][batch_order]
                Y_train_cp = Y_train[i_train][batch_order, ]
                # Solve epoch
                for batch_i in range(0, X_train_cp[i_train].shape[0], batch_size):
                    if (batch_i + batch_size) < X_train_cp[i_train].shape[0]:
                        train_models[i_train].train_on_batch(X_train_cp[i_train][batch_i: batch_i + batch_size],
                                                             Y_train_cp[batch_i: batch_i + batch_size],
                                                             accuracy=True)
                    else:
                        train_models[i_train].train_on_batch(X_train_cp[i_train][batch_i:],
                                                             Y_train_cp[batch_i:],
                                                             accuracy=True)
                score = train_models[i_train].evaluate(X_train[i_train], Y_train[i_train],
                                                       verbose=0, show_accuracy=True)
                print('For batch %d: train score: %.2f, train accuracy: %.3f' % (i_train, score[0], score[1]))
                score = train_models[i_train].evaluate(X_test, Y_test, verbose=0, show_accuracy=True)
                print('For batch %d: test score: %.2f, test accuracy: %.3f' % (i_train, score[0], score[1]))
            # Fit calculated model to the test data
            batch_predict_test = []
            for i_train in range(n_ensemble):
                batch_predict_test.append(train_models[i_train].predict_proba(X_test,
                                                                              batch_size=batch_size,
                                                                              verbose=1))
            batch_predicted_results = np.zeros(batch_predict_test[0].shape)
            for mat in batch_predict_test:
                batch_predicted_results += mat
                batch_predicted_results /= n_ensemble
            print('The average test score %.3f' % log_loss(train_labels[test_cv_ind], batch_predicted_results))
        test_predicted_results = []
        for i_train in range(n_ensemble):
            test_predicted_results.append(train_models[i_train].predict_proba(test_files_cnn,
                                                                              batch_size=batch_size,
                                                                              verbose=1))
        """
        Get accuracy
        """
        # predicted_results = model.predict_classes(X_test, batch_size=batch_size, verbose=1)
        # print(label_encoder.inverse_transform(predicted_results))
        # print(label_encoder.inverse_transform(y_test))

"""
Solve and submit test
"""
# print('The Estimated Log loss is %f ' % np.mean(test_acc))

sub_file = pd.DataFrame.from_csv('sample_submission.csv')

predicted_results = np.zeros(sub_file.shape)
for mat in test_predicted_results:
    predicted_results += mat
predicted_results /= len(test_predicted_results)
print(predicted_results)

sub_file.iloc[:, :] = predicted_results
sub_file = sub_file.fillna(0.1)

# Ordering sample index when needed
test_index = []
for file_name in test_names:
    test_index.append(file_name.split('/')[-1])
sub_file.index = test_index
sub_file.index.name = 'img'

sub_file.to_csv(submit_name)
#
# # no image preprocessing:
