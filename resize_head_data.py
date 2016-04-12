import glob
from skimage.transform import resize
from skimage.io import imread, imsave
import os

# Set path of data files
path = "imgs"

if not os.path.exists(path + "/trainResizedHead"):
    os.makedirs(path + "/trainResizedHead")
if not os.path.exists(path + "/testResizedHead"):
    os.makedirs(path + "/testResizedHead")

img_size_y = 50
img_size_x = 80

train_files_unlabeled = sorted(glob.glob(path + "/train/*"))
for fol in train_files_unlabeled:
    if not os.path.exists(path + "/trainResizedHead/" + fol.split("/")[-1]):
        os.makedirs(path + "/trainResizedHead/" + fol.split("/")[-1])

train_files_labeled = []
for fol in train_files_unlabeled:
    train_files_labeled.append(glob.glob(fol + '/*'))

for fol in train_files_labeled:
    for im_name in fol:
        image = imread(im_name)
        image = image[50:300, :400, :]
        imageResized = resize(image, (img_size_y, img_size_x))
        newName = "/".join(im_name.split("/")[:-2]) + "ResizedHead/" + "/".join(im_name.split("/")[-2:])
        imsave(newName, imageResized)

testFiles = sorted(glob.glob(path + "/test/*"))
for i, nameFile in enumerate(testFiles):
    image = imread(nameFile)
    imageResized = resize(image, (img_size_y, img_size_x))
    newName = "/".join(nameFile.split("/")[:-1]) + "ResizedHead/" + nameFile.split("/")[-1]
    imsave(newName, imageResized)
