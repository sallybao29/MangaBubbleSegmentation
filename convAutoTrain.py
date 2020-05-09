import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import pickle
import cv2
import numpy as np

VALIDATION_SPLIT = 0.2
EPOCHS = 5
BATCH_SIZE = 64
NUM_VALID = None
NUM_INVALID = None
IMAGE_SHAPE = (256, 256, 1)

path = os.getcwd()
pathStorage = os.path.join(path, "Models", "Auto")
pathPickle = os.path.join(pathStorage, "data.p")

def transform(image):
    image = np.expand_dims(np.asarray(image), axis=2)
    image = np.array(image)/255
    return image
    
def loadfiles(path, valid=None, invalid=None): 
    data = []
    validTrain = os.path.join(path, "TrainingValid")
    invalidTrain = os.path.join(path, "TrainingInvalid")
    validLabelled = os.path.join(path, "LabelledValid")
    invalidLabelled = os.path.join(path, "LabelledInvalid")
    counter = 0
    for image in os.listdir(validTrain):
        validImage = cv2.imread(os.path.join(validTrain, image), cv2.IMREAD_GRAYSCALE)
        validLabel = cv2.imread(os.path.join(validLabelled, image), cv2.IMREAD_GRAYSCALE)
        data.append((transform(validImage), transform(validLabel)))
        counter += 1
        if counter % 10000 == 0:
            print("loaded %d images"%counter)
        if valid and counter == valid:
            break
    counter2 = 0
    for image in os.listdir(invalidTrain):
        invalidImage = cv2.imread(os.path.join(invalidTrain, image), cv2.IMREAD_GRAYSCALE)
        invalidLabel = cv2.imread(os.path.join(invalidLabelled, image), cv2.IMREAD_GRAYSCALE)
        data.append((transform(invalidImage), transform(invalidLabel)))
        counter += 1
        counter2 += 1
        if counter % 10000 == 0:
            print("loaded %d images"%counter)
        if invalid and counter2 == invalid:
            break
    return np.array(data)
        
def imageGenerator(samples, batchSize):
    numSamples = len(samples)
    while True:
        batch = np.random.permutation(samples)[0:batchSize] 
        images = []
        masks = []
        for b in batch:
            images.append(b[0])
            masks.append(b[1])
        yield (np.array(images), np.array(masks))
        
if (os.path.exists(pathPickle)):
    with open(pathPickle, "rb") as fp:
        trainSamples, testSamples = pickle.load(fp)
else:
    files = loadfiles(path, NUM_VALID, NUM_INVALID)
    perm = np.random.permutation(files)
    num_files = len(files)
    num_train = int(num_files*(1-VALIDATION_SPLIT))
    trainSamples = perm[:num_train]
    testSamples = perm[num_train:]
    
    print("Loaded files")
    with open(pathPickle, "wb") as fp:
        pickle.dump([trainSamples,testSamples], fp)

ntr = len(trainSamples)
nts = len(testSamples)

train_generator = imageGenerator(trainSamples, batchSize=BATCH_SIZE)
validation_generator = imageGenerator(trainSamples, batchSize=BATCH_SIZE)

input_img = keras.Input(shape=IMAGE_SHAPE)

x = layers.Conv2D(16, (3, 3), activation='relu', padding='same', data_format="channels_last")(input_img)
x = layers.MaxPooling2D((2, 2), padding='same')(x)
x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = layers.MaxPooling2D((2, 2), padding='same')(x)
x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
encoded = layers.MaxPooling2D((2, 2), padding='same')(x)

x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
x = layers.UpSampling2D((2, 2))(x)
x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = layers.UpSampling2D((2, 2))(x)
x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
x = layers.UpSampling2D((2, 2))(x)
decoded = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = keras.Model(input_img, decoded)
#autoencoder.summary()

autoencoder.compile(optimizer='adam',
                    loss="binary_crossentropy",
                    metrics=['accuracy'])

autoencoder.fit_generator(
    train_generator,
    steps_per_epoch=ntr//BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=validation_generator,
    validation_steps=nts//BATCH_SIZE)

autoencoder.save(os.path.join(pathStorage, "0.h5"))


            
