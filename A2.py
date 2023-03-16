#start to learn about image
import tensorflow as tf

from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import pandas as pd
from PIL import Image
import glob
#load the image
#convert to array
#set GPU
tf.config.experimental.list_physical_devices('GPU')

imexample = cv2.imread('D:/21-22 sem2/Big data/images/images/42.6804_2.9496714.png')
(h,w) = imexample.shape[:2]

#url = 'D:/21-22 sem2/Big data/images/images/42.6804_2.9496714.png'
#print(url[37:])

def loadimage(location):
    image_list = []
    name_list = []
    for filename in glob.glob(location):
    #print(filename)
       im = load_img(filename,target_size = (224, 224))
       imarray = img_to_array(im)
       image_list.append(imarray)
       im.close()
       pngname = filename[37:]
       name_list.append(pngname)
    return (image_list,name_list)

(allimage,allname) = loadimage('D:/21-22 sem2/Big data/images/images/*.png')


#get bounding box
import json

with open('D:/21-22 sem2/Big data/metadata1.json') as data_file:    
    labels = json.load(data_file)

#get bounding box
def converttarget(allkey):
    allrectangle = []
    piclist = []
    for pic in allkey:
        picinfo = labels[pic]["bounds_x_y"]
        piclist.append(pic)
        xlist = []
        ylist= []
        for i in range(len(picinfo)):
            x = picinfo[i]['x']
            y = picinfo[i]['y']
            xlist.append(x)
            ylist.append(y)
            box = (min(xlist)/w,min(ylist)/h,max(xlist)/w,max(ylist)/h)
        allrectangle.append(box)

    #targetset = pd.DataFrame()
    #targetset['image'] = allkeys
    #targetset['box'] = allrectangle

    return (allrectangle,piclist)


allkeys = labels.keys()
boxes,piclist = converttarget(allkeys)

#match boxes and pictures
pics = []
bbox = []
for i in range(len(allname)):
    for j in range(len(piclist)):
        if allname[i] == piclist[j]:
            pics.append(allimage[i])
            bbox.append(boxes[j])



from shapely.geometry import MultiPolygon, Polygon, LineString

#get polygon
allkeys = labels.keys()
allpolygon = []
for pic in allkeys:
    picinfo = labels[pic]["bounds_x_y"]
    polygonlist = []
    for i in range(len(picinfo)):
        points = (picinfo[i]['x'],picinfo[i]['y'])
        polygonlist.append(points)
    allpolygon.append(Polygon(polygonlist))


# convert the data and targets to NumPy arrays, scaling the input
# pixel intensities from the range [0, 511] to [0, 1]

data = np.array(pics, dtype="float32") / 511.0
targets = np.array(bbox, dtype="float32")

# partition the data into training and testing splits using 90% of
# the data for training and the remaining 10% for testing
split = train_test_split(data, targets, test_size=0.20,
	random_state=42)
# unpack the data split
(trainImages, testImages) = split[:2]
(trainTargets, testTargets) = split[2:4]
#(trainFilenames, testFilenames) = split[4:]
# write the testing filenames to disk so that we can use then
# when evaluating/testing our bounding box regressor
#print("[INFO] saving testing filenames...")
#f = open(config.TEST_FILENAMES, "w")
#f.write("\n".join(testFilenames))
#f.close()

# load the VGG16 network, ensuring the head FC layers are left off
vgg = VGG16(weights="imagenet", include_top=False,
	input_tensor=Input(shape=(224, 224, 3)))
# freeze all VGG layers so they will *not* be updated during the
# training process
vgg.trainable = False
# flatten the max-pooling output of VGG
flatten = vgg.output
flatten = Flatten()(flatten)
# construct a fully-connected layer header to output the predicted
# bounding box coordinates
bboxHead = Dense(128, activation="relu")(flatten)
bboxHead = Dense(64, activation="relu")(bboxHead)
bboxHead = Dense(32, activation="relu")(bboxHead)
bboxHead = Dense(4, activation="sigmoid")(bboxHead)
# construct the model we will fine-tune for bounding box regression
model = Model(inputs=vgg.input, outputs=bboxHead)

# initialize the optimizer, compile the model, and show the model
# summary

INIT_LR = 1e-4
NUM_EPOCHS = 25
BATCH_SIZE = 32

opt = Adam(lr=INIT_LR)
model.compile(loss="mse", optimizer=opt)
print(model.summary())
# train the network for bounding box regression
print("[INFO] training bounding box regressor...")
H = model.fit(
	trainImages, trainTargets,
	validation_data=(testImages, testTargets),
	batch_size=BATCH_SIZE,
	epochs=NUM_EPOCHS,
	verbose=1)

H
N = NUM_EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.title("Bounding Box Regression Loss on Training Set")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(loc="lower left")

H.history["loss"]


