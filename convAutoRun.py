import os
import pickle
import cv2
import numpy as np
import tensorflow as tf

path = os.getcwd()
pathStorage = os.path.join(path, "Models", "Auto")
pathPickle = os.path.join(pathStorage, "data.p")
pathModel = os.path.join(pathStorage, "0.h5")
pathTrainValid = os.path.join(path, "TrainingValid")
pathLabelledValid = os.path.join(path, "LabelledValid")

"""
def display(display_list):
  plt.figure(figsize=(15, 15))

  title = ['Input Image', 'True Mask', 'Predicted Mask']

  for i in range(len(display_list)):
    plt.subplot(1, len(display_list), i+1)
    plt.title(title[i])
    plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
    plt.axis('off')
  plt.show()
"""

def normalize(image):
    image = np.expand_dims(np.asarray(image), axis=2)
    image = np.array(image)/255
    return image

def denormalize(image):
    image *= 255
    return image.astype(np.uint8)

def predict_mask(model, sample, save_path=None):
    image, true_mask = sample
    # model expects image with shape (samples, height, width, channels)
    pred_mask = model.predict(image[None,:])[0]
    pred_mask = denormalize(pred_mask)
    true_mask = denormalize(true_mask)
    #if plot:
       #display([image, true_mask, pred_mask])
    if save_path:
        cv2.imwrite(save_path+"true.jpg", true_mask)
        cv2.imwrite(save_path+"pred.jpg", pred_mask)

model = tf.keras.models.load_model(pathModel)

#with open(pathPickle, "rb") as fp:
    #trainSamples, testSamples = pickle.load(fp) 

#predict_mask(model, trainSamples[0], save_path=os.path.join(pathStorage, "test"))

image = cv2.imread(os.path.join(pathTrainValid, "1.jpg"), cv2.IMREAD_GRAYSCALE)
true_mask = cv2.imread(os.path.join(pathLabelledValid, "1.jpg"), cv2.IMREAD_GRAYSCALE)

predict_mask(model, (normalize(image), normalize(true_mask)), save_path=os.path.join(pathStorage, "test"))
