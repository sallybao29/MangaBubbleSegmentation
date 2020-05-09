import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

path = os.getcwd()

pathTrain = os.path.join(path, "TrainingConv")
pathValidation = os.path.join(path, "ValidationConv")
pathStorage = os.path.join(path, "Models", "Convo")

model = models.Sequential()
model.add(layers.Conv2D(8, (3, 3), padding="same", activation='relu', input_shape=(256, 256, 1), data_format="channels_last"))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(8, (3, 3), padding="same", activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(8, (3, 3), padding="same", activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

#filename = len(os.listdir(pathStorage))
#filename = 4
#pathCheckpoint = os.path.join(pathStorage, str(filename - 1)+ ".ckpt")
#pathOutput = os.path.join(pathStorage, str(filename)+ ".ckpt")
#model.load_weights(pathCheckpoint)

#cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath = pathCheckpoint,
#                                                 save_weights_only=True,
#                                                 verbose=1)

def preprocessing(image):
    return (image/255.0)-1.0

datagen = ImageDataGenerator(preprocessing_function = preprocessing)
dataTrain = datagen.flow_from_directory(directory = pathTrain,
                                        target_size=(256,256),
                                        color_mode="grayscale",
                                        class_mode="binary",
                                        batch_size = 32,
                                        shuffle=False)
"""
dataTest = datagen.flow_from_directory(directory = pathValidation,
                                        target_size=(256,256),
                                        color_mode="grayscale",
                                        class_mode="binary",
                                        batch_size = 100,
                                        shuffle=False)
"""

model.compile(optimizer='adam',
              loss="binary_crossentropy",
              metrics=['accuracy'])

for i in range(10):
    pathOutput = os.path.join(pathStorage, str(i)+ ".ckpt")
    model.fit_generator(dataTrain, 50000, 1) #, callbacks=[cp_callback])
    model.save(pathOutput)
