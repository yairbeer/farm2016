import glob
from skimage.transform import resize
from skimage.io import imread, imsave
import os

# Set path of data files
path = "imgs"

if not os.path.exists(path + "/trainResized"):
    os.makedirs(path + "/trainResized")
if not os.path.exists(path + "/testResized"):
    os.makedirs(path + "/testResized")

img_size = 40

train_files_unlabeled = sorted(glob.glob(path + "/train/*"))
for fol in train_files_unlabeled:
    if not os.path.exists(path + "/trainResized/" + fol.split("/")[-1]):
        os.makedirs(path + "/trainResized/" + fol.split("/")[-1])

train_files_labeled = []
for fol in train_files_unlabeled:
    train_files_labeled.append(glob.glob(fol + '/*'))

for fol in train_files_labeled:
    for im_name in fol:
        image = imread(im_name)
        imageResized = resize(image, (img_size, img_size))
        newName = "/".join(im_name.split("/")[:-2]) + "Resized/" + "/".join(im_name.split("/")[-2:])
        imsave(newName, imageResized)

testFiles = sorted(glob.glob(path + "/test/*"))
for i, nameFile in enumerate(testFiles):
    image = imread(nameFile)
    imageResized = resize(image, (img_size, img_size))
    newName = "/".join(nameFile.split("/")[:-1]) + "Resized/" + nameFile.split("/")[-1]
    imsave(newName, imageResized)
