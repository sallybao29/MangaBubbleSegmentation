import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import cv2
import numpy as np

path = os.getcwd()
pathStorage = os.path.join(path, "Models", "Convo")

pathInput = os.path.join(path, "Grayscale")
pathOutput = os.path.join(path, "HeatmapResults", "Convo")

windowSize = 256
stepSize = windowSize // 8


def genHeatMap(model, inputpath, outputpath):
    for manga in os.listdir(inputpath):
        mangaPathIn = os.path.join(inputpath, manga)
        mangaPathOut = os.path.join(outputpath, manga)
        os.mkdir(mangaPathOut)
        for chapter in os.listdir(mangaPathIn):
            chapterPathIn = os.path.join(mangaPathIn, chapter)
            chapterPathOut = os.path.join(mangaPathOut, chapter)
            os.mkdir(chapterPathOut)
            for page in os.listdir(chapterPathIn):
                img = cv2.imread(os.path.join(chapterPathIn, page), cv2.IMREAD_GRAYSCALE)
                img = (img / 255 - 1)

                rows, cols = img.shape
                heatMap = np.zeros(img.shape);
                seg = np.zeros((1,256,256,1))

                dim = min(img.shape) // windowSize
                maxDivs = 1
                while dim > 2:
                    dim //= 2
                    maxDivs += 1

                for i in range(maxDivs):
                    mult = (2 ** i)
                    dim = (int(rows) // mult, cols // mult)
                    imgScaled = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

                    bufferX = (stepSize - (cols % stepSize)) % stepSize
                    bufferY = (stepSize - (rows % stepSize)) % stepSize

                    imgScaled = cv2.copyMakeBorder(imgScaled, 0, bufferY, 0, bufferX, cv2.BORDER_REPLICATE)

                    for r in range(imgScaled.shape[0] // stepSize - 7):
                        for c in range(imgScaled.shape[1] // stepSize - 7):
                            left = c * stepSize
                            bottom = r * stepSize

                            seg[0] = np.expand_dims(imgScaled[bottom:bottom + windowSize, left:left + windowSize], -1)
                            res = model.predict(seg)[0][0]
                            #print(res)
                            res /= mult
                            heatMap[left * mult: (left + windowSize) * mult,
                            bottom * mult: (bottom + windowSize) * mult] += res
                heatMap /= np.max(heatMap)
                heatMap *= 255
                cv2.imwrite(os.path.join(chapterPathOut, page), heatMap.astype(np.uint8))
                print(os.path.join(chapterPathOut, page))


model = tf.keras.models.load_model(os.path.join(pathStorage, "2.ckpt"))
genHeatMap(model, pathInput, pathOutput)
