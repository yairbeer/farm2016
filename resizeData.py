import glob
from skimage.transform import resize
from skimage.io import imread, imsave
import os

# Set path of data files
path = "imgs"

if not os.path.exists(path + "/trainResized40"):
    os.makedirs(path + "/trainResized40")
if not os.path.exists(path + "/testResized40"):
    os.makedirs(path + "/testResized40")

img_size_y = 40
img_size_x = 40

train_files_unlabeled = sorted(glob.glob(path + "/train/*"))
for fol in train_files_unlabeled:
    if not os.path.exists(path + "/trainResized40/" + fol.split("/")[-1]):
        os.makedirs(path + "/trainResized40/" + fol.split("/")[-1])

train_files_labeled = []
for fol in train_files_unlabeled:
    train_files_labeled.append(glob.glob(fol + '/*'))

for fol in train_files_labeled:
    for im_name in fol:
        image = imread(im_name)
        imageResized = resize(image, (img_size_y, img_size_x))
        newName = "/".join(im_name.split("/")[:-2]) + "Resized40/" + "/".join(im_name.split("/")[-2:])
        imsave(newName, imageResized)

testFiles = sorted(glob.glob(path + "/test/*"))
for i, nameFile in enumerate(testFiles):
    image = imread(nameFile)
    imageResized = resize(image, (img_size_y, img_size_x))
    newName = "/".join(nameFile.split("/")[:-1]) + "Resized40/" + nameFile.split("/")[-1]
    imsave(newName, imageResized)
