import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np
import cv2

path = os.getcwd()
pathStorage = os.path.join(path, "Models", "Seg")
pathGray = (os.path.join(path, "TrainingValid"), os.path.join(path, "TrainingInvalid"))
pathMask = (os.path.join(path, "LabelledValid"), os.path.join(path, "LabelledInvalid"))


img_input = layers.Input(shape=(256, 256, 1))
conv1_1 = layers.Conv2D(8, (3, 3), padding="same", activation='relu')(img_input)
conv1_2 = layers.Conv2D(8, (3, 3), padding="same", activation='relu')(conv1_1)
pool1 = layers.MaxPooling2D((2, 2))(conv1_2)

conv2_1 = layers.Conv2D(8, (3, 3), padding="same", activation='relu')(pool1)
conv2_2 = layers.Conv2D(8, (3, 3), padding="same", activation='relu')(conv2_1)
pool2 = layers.MaxPooling2D((2, 2))(conv2_2)

conv3_1 = layers.Conv2D(8, (3, 3), padding="same", activation='relu')(pool2)
conv3_2 = layers.Conv2D(8, (3, 3), padding="same", activation='relu')(conv3_1)
pool3 = layers.MaxPooling2D((2, 2))(conv3_2)

conv4_1 = layers.Conv2D(8, (3, 3), padding="same", activation='relu')(pool3)
conv4_2 = layers.Conv2D(8, (3, 3), padding="same", activation='relu')(conv4_1)
pool4 = layers.MaxPooling2D((2, 2))(conv4_2)

conv5_1 = layers.Conv2D(8, (3, 3), padding="same", activation='relu')(pool4)
conv5_2 = layers.Conv2D(8, (3, 3), padding="same", activation='relu')(conv5_1)
pool5 = layers.MaxPooling2D((2, 2))(conv5_2)

prenet = layers.Flatten()(layers.Conv2D(1, (1,1), activation='sigmoid')(pool5))
flat1 = layers.Dense(64, activation='relu')(prenet)
flat2 = layers.Dense(64, activation='relu')(flat1)
flat3 = layers.Dense(64, activation='relu')(flat2)
postnet = layers.Reshape((8,8,1))(flat3)

concat1 = layers.concatenate([layers.UpSampling2D((2,2))(postnet), conv5_2], -1)
uconv1_1 = layers.Conv2D(8, (3, 3), padding="same", activation='relu')(concat1)
uconv1_2 = layers.Conv2D(8, (3, 3), padding="same", activation='relu')(uconv1_1)

concat2 = layers.concatenate([layers.UpSampling2D((2,2))(uconv1_2), conv4_2], -1)
uconv2_1 = layers.Conv2D(8, (3, 3), padding="same", activation='relu')(concat2)
uconv2_2 = layers.Conv2D(8, (3, 3), padding="same", activation='relu')(uconv2_1)

concat3 = layers.concatenate([layers.UpSampling2D((2,2))(uconv2_2), conv3_2], -1)
uconv3_1 = layers.Conv2D(8, (3, 3), padding="same", activation='relu')(concat3)
uconv3_2 = layers.Conv2D(8, (3, 3), padding="same", activation='relu')(uconv3_1)

concat4 = layers.concatenate([layers.UpSampling2D((2,2))(uconv3_2), conv2_2], -1)
uconv4_1 = layers.Conv2D(8, (3, 3), padding="same", activation='relu')(concat4)
uconv4_2 = layers.Conv2D(8, (3, 3), padding="same", activation='relu')(uconv4_1)

concat5 = layers.concatenate([layers.UpSampling2D((2,2))(uconv4_2), conv1_2], -1)
uconv5_1 = layers.Conv2D(8, (3, 3), padding="same", activation='relu')(concat5)
uconv5_2 = layers.Conv2D(8, (3, 3), padding="same", activation='relu')(uconv5_1)

img_output = layers.Conv2D(1, (1,1), activation='sigmoid')(uconv5_2)

model = models.Model(img_input, img_output)

model.summary()

def batches(pathGrays, pathMasks, batch_size, steps):
    data = np.zeros((batch_size,256,256,1))
    mask = np.zeros((batch_size,256,256,1))
    for folder in range(len(pathGrays)):
        counter = 0
        paths = os.listdir(pathGrays[folder])[:batch_size * steps[folder]]
        for i in range(batch_size * steps[folder]):
            data[counter] = np.expand_dims(cv2.imread(os.path.join(pathGrays[folder], paths[i]), cv2.IMREAD_GRAYSCALE), -1)
            mask[counter] = np.expand_dims(cv2.imread(os.path.join(pathMasks[folder], paths[i]), cv2.IMREAD_GRAYSCALE), -1)
            counter += 1
            if counter == batch_size:
                counter = 0
                yield (data/128.0 - 1.0), (mask/255.0)

model.compile(optimizer='adam',
              loss="binary_crossentropy",
              metrics=['accuracy'])

for i in range(1,6):
    counter = 0
    # should be 75000 instead of 45000
    for x, y in batches(pathGray, pathMask, 32, [5000, 75000]):
        model.train_on_batch(x,y)
        counter+=1
        if counter % 800 == 0:
            print(counter//800,"%")
        if counter % 8000 == 0:
            model.save(os.path.join(pathStorage, str(i) + "_" + str(counter // 8000) + ".ckpt"))
            print("Saved", str(i) + "_" + str(counter // 8000) + ".ckpt")
    sumVal = [0.0, 0.0]
    counter = 0
    for x, y in batches(pathGray, pathMask, 32, [5000, 75000]):
        tmp = model.evaluate(x, y, verbose = 0)
        sumVal[0] += tmp[0]
        sumVal[1] += tmp[1]
    print(sumVal[0]/80000.0, sumVal[1]/80000.0)

