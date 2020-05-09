import pickle
import numpy as np
import scipy
from sklearn import linear_model
import cv2
import os

path = os.getcwd()
pathStorage = os.path.join(path, "Models", "Log")
pathInput = os.path.join(path, "Grayscale")
pathOutput = os.path.join(path, "HeatmapResults", "Log")
pathSource = (os.path.join(path, "TrainingValid"), os.path.join(path, "TrainingInvalid"))

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
                img = (img/255 - 0.5)
                
                rows, cols = img.shape
                heatMap = np.zeros(img.shape);

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
                            left = c*stepSize
                            bottom = r*stepSize

                            seg = np.ravel(imgScaled[bottom:bottom + windowSize, left:left + windowSize])
                            seg = seg.reshape(1, -1)
                            #print(seg.shape, bottom, bottom + windowSize, left, left + windowSize)
                            res = model.predict(seg)[0] / mult

                            heatMap[left*mult : (left+windowSize)*mult, bottom*mult : (bottom+windowSize)*mult] += res
                heatMap /= np.max(heatMap)
                heatMap *= 255
                cv2.imwrite(os.path.join(chapterPathOut, page), heatMap.astype(np.uint8))
                print(os.path.join(chapterPathOut, page))

for save in os.listdir(pathStorage):
    pickleOutput = os.path.join(pathStorage, save)
    logreg = linear_model.LogisticRegression(penalty='l2', solver='newton-cg', max_iter=500)
    with open(pickleOutput, "rb" ) as fp:
        logreg = pickle.load(fp)
        currOutput = os.path.join(pathOutput, save.replace(".p", ""))
        if not os.path.isdir(currOutput):
            os.mkdir(currOutput)
        else:
            continue
        genHeatMap(logreg, pathInput, currOutput)
