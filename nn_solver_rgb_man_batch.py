import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.io import imread
from sklearn.preprocessing import LabelEncoder
from skimage.color import gray2rgb
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
        plt.imshow(img)
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
    # if gray convert to color
    if len(img.shape) == 2:
        img = gray2rgb(img)
    return img


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
submit_name = 'rgb_32x24_man.csv'
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

train_files = np.zeros((len(train_names), img_size_y, img_size_x, 3)).astype('float32')
train_labels = np.zeros((len(train_names),)).astype(str)
for i, name_file in enumerate(train_names):
    image = imp_img(name_file)
    train_files[i, :, :, :] = image
    train_labels[i] = name_file.split('/')[-2]

# Test
test_names = sorted(glob.glob(path + "/testResizedSmall/*"))
test_files = np.zeros((len(test_names), img_size_y, img_size_x, 3)).astype('float32')
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

"""
Configure train/test by drivers and images per state
"""
n_fold = 10
percent_drivers = 0.5
imgs_per_driver = 1000

batch_size = 128
nb_classes = 10
nb_epoch = 10
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
train_files_cnn = np.zeros((train_files.shape[0], 3, img_rows, img_cols)).astype('float32')
test_files_cnn = np.zeros((test_files.shape[0], 3, img_rows, img_cols)).astype('float32')

for i in range(train_files_cnn.shape[0]):
    train_files_cnn[i, 0, :, :] = train_files[i, :, :, 0]
    train_files_cnn[i, 1, :, :] = train_files[i, :, :, 1]
    train_files_cnn[i, 2, :, :] = train_files[i, :, :, 2]

for i in range(test_files_cnn.shape[0]):
    test_files_cnn[i, 0, :, :] = test_files[i, :, :, 0]
    test_files_cnn[i, 1, :, :] = test_files[i, :, :, 1]
    test_files_cnn[i, 2, :, :] = test_files[i, :, :, 2]

# convert class vectors to binary class matrices
train_labels_dummy = np_utils.to_categorical(train_labels, nb_classes)

test_results = []
test_acc = []
for i_fold in range(n_fold):
    # Get all the drivers
    drivers_index = np.unique(drivers.index.values)

    # Seed for repeatability
    np.random.seed(1000 * i_fold ** 2)
    train_test_driver_index = np.random.choice(range(drivers_index.shape[0]), drivers_index.shape[0], replace=False)
    train_driver_index = train_test_driver_index[: int(drivers_index.shape[0] * percent_drivers)]
    test_driver_index = train_test_driver_index[int(drivers_index.shape[0] * percent_drivers):]
    # On Average the number of drivers is cv_prob percent of the data
    train_cv_drivers = drivers_index[train_driver_index]

    train_cv_ind = np.zeros((train_files.shape[0],)).astype(bool)
    test_cv_ind = np.zeros((train_files.shape[0],)).astype(bool)

    train_images = []
    # For each driver
    for driver in train_cv_drivers:
        driver_imgs = drivers.loc[train_cv_drivers]
        avail_states = np.unique(driver_imgs.classname.values)
        # For each driving state
        for state in avail_states:
            # Get imgs_per_driver images (using all the images can overfit)
            driver_state_imgs = driver_imgs.iloc[np.array(driver_imgs.classname == state)].img.values
            if imgs_per_driver < driver_state_imgs.shape[0]:
                train_img_index = np.random.choice(driver_state_imgs.shape[0], imgs_per_driver, replace=False)
                train_images += list(driver_state_imgs[train_img_index])
            else:
                train_images += list(driver_state_imgs)
    train_images = np.array(train_images)

    test_images = []
    # Use all images of the test driver as test
    test_cv_drivers = drivers_index[test_driver_index]
    for driver in test_cv_drivers:
        test_images += list(drivers.loc[driver].img.values)
    test_images = np.array(test_images)

    for i, file_name in enumerate(train_names):
        img_name = file_name.split('/')[-1]
        if img_name in train_images:
            train_cv_ind[i] = True
        if img_name in test_images:
            test_cv_ind[i] = True

    # Get the train / test split
    X_train, Y_train = train_files_cnn[train_cv_ind].astype('float32'), train_labels_dummy[train_cv_ind, :]
    X_test, Y_test = train_files_cnn[test_cv_ind].astype('float32'), train_labels_dummy[test_cv_ind, :]

    """
    Compile Model
    """
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    np.random.seed(100 * i_fold ** 3)  # for reproducibility

    """
    CV model
    """
    model = Sequential()
    model.add(Convolution2D(nb_filters, nb_conv, nb_conv,
                            border_mode='valid', input_shape=(3, img_rows, img_cols)))
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

    for epoch_i in range(nb_epoch):
        X_train_cp = np.array(X_train, copy=True)
        print('Epoch %d' % epoch_i)
        if epoch_i in lr_updates:
            print('lr changed to %f' % lr_updates[epoch_i])
            model.optimizer.lr.set_value(lr_updates[epoch_i])
        np.random.seed(epoch_i)
        rotate_angle = np.random.normal(0, 3, X_train_cp.shape[0])
        rescale_fac = np.random.normal(1, 0.05, X_train_cp.shape[0])
        right_move = np.random.normal(0, 0.1, X_train_cp.shape[0])
        up_move = np.random.normal(0, 0.1, X_train_cp.shape[0])
        shear = np.random.normal(0, 5, X_train_cp.shape[0])
        shear = np.deg2rad(shear)
        for img_i in range(X_train_cp.shape[0]):
            afine_tf = tf.AffineTransform(shear=shear[img_i])
            for color in range(3):
                X_train_cp[img_i, color, :, :] = tf.warp(X_train_cp[img_i, color, :, :], afine_tf)
                X_train_cp[img_i, color, :, :] = img_rotate(X_train_cp[img_i, color, :, :], rotate_angle[img_i], -1)
                X_train_cp[img_i, color, :, :] = img_rescale(X_train_cp[img_i, color, :, :], rescale_fac[img_i])
                X_train_cp[img_i, color, :, :] = img_leftright(X_train_cp[img_i, color, :, :], right_move[img_i])
                X_train_cp[img_i, color, :, :] = img_updown(X_train_cp[img_i, color, :, :], up_move[img_i])
        batch_order = np.random.choice(range(X_train_cp.shape[0]), X_train_cp.shape[0], replace=False)
        X_train_cp = X_train_cp[batch_order, :, :, :]
        Y_train_cp = Y_train[batch_order, :]
        for batch_i in range(0, X_train_cp.shape[0], batch_size):
            if (batch_i + batch_size) < X_train_cp.shape[0]:
                model.train_on_batch(X_train_cp[batch_i: batch_i + batch_size],
                                     Y_train_cp[batch_i: batch_i + batch_size],
                                     accuracy=True)
            else:
                model.train_on_batch(X_train_cp[batch_i:], Y_train_cp[batch_i:], accuracy=True)
        score = model.evaluate(X_train, Y_train, verbose=0, show_accuracy=True)
        print('Train score: %.2f, Train accuracy: %.3f' % (score[0], score[1]))
        score = model.evaluate(X_test, Y_test, verbose=0, show_accuracy=True)
        print('Test score: %.2f, Test accuracy: %.3f' % (score[0], score[1]))
    """
    Get accuracy
    """
    # predicted_results = model.predict_classes(X_test, batch_size=batch_size, verbose=1)
    # print(label_encoder.inverse_transform(predicted_results))
    # print(label_encoder.inverse_transform(y_test))
    test_acc.append(score[0])
    predicted_results = model.predict_proba(test_files_cnn, batch_size=batch_size, verbose=1)
    test_results.append(predicted_results)
"""
Solve and submit test
"""
print('The Estimated Log loss is %f ' % np.mean(test_acc))

sub_file = pd.DataFrame.from_csv('sample_submission.csv')

predicted_results = np.zeros(sub_file.shape)
for mat in test_results:
    predicted_results += mat
predicted_results /= len(test_results)
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

# no image preprocessing:
