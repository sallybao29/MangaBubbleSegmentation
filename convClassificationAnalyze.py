import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

path = os.getcwd()

pathTrain = os.path.join(path, "TrainingConv")
pathValidation = os.path.join(path, "ValidationConv")
pathStorage = os.path.join(path, "Models", "Convo")

modelCount = 10

def preprocessing(image):
    return (image/255.0)-1.0

datagen = ImageDataGenerator(preprocessing_function = preprocessing)

dataTest = datagen.flow_from_directory(directory = pathValidation,
                                        target_size=(256,256),
                                        color_mode="grayscale",
                                        class_mode="binary",
                                        batch_size = 32,
                                        shuffle=False)

for i in range(modelCount):
    model = tf.keras.models.load_model(os.path.join(pathStorage, str(i) + ".ckpt"))
    print(model.evaluate(dataTest, steps = 35000))
