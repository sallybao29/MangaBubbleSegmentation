import os
import cv2
import numpy as np
import tensorflow as tf
import sys

pathModel = "1_3.ckpt"

def transform(image):
    image = np.expand_dims(np.asarray(image), axis=2)
    image = np.array(image)/128.0-1
    return image

def denormalize(image):
  image *= 255
  return image.astype(np.uint8)

def predict_mask(model, image, true_mask=None, save_path=None):
    # model expects image with shape (samples, height, width, channels)
    im = transform(image)
    pred_mask = model.predict(im[None,:])[0]
    pred_mask = denormalize(pred_mask)
    
    cv2.imshow("pred", pred_mask)
    cv2.waitKey()
    
    if save_path:
        cv2.imwrite("pred"+save_path, pred_mask)

def main(argv):
    if len(argv) != 1:
        return

    imagePath = argv[0]
    model = tf.keras.models.load_model(pathModel)
    image = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)
    predict_mask(model, image, save_path=argv[0])

    
main(sys.argv[1:])
