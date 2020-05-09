import os
import numpy as np
import cv2
import imutils

#path = os.path.join("/content", "drive","My Drive","Machine Learning Project")
#path = ""
path = os.getcwd()
#print(path)
pathManga = os.path.join(path,"Original")
pathGray = os.path.join(path,"Grayscale")
pathLabelled = os.path.join(path,"Training")
pathOutput = (os.path.join(path, "TrainingValid"), os.path.join(path, "TrainingInvalid"), os.path.join(path, "LabelledValid"), os.path.join(path, "LabelledInvalid"))

windowSize = 256
stepSize = windowSize // 2

lr1 = np.array([0,191,223])
ur1 = np.array([7,255,255])
lr2 = np.array([173,191,223])
ur2 = np.array([180,255,255])

ly1 = np.array([23,191,223])
uy1 = np.array([37,255,255])

def getLabelSegmentation(imgLabelled):
  hsv = cv2.cvtColor(imgLabelled, cv2.COLOR_BGR2HSV)
  maskR = cv2.inRange(hsv, lr1, ur1) + cv2.inRange(hsv, lr2, ur2)
  maskY = cv2.inRange(hsv, ly1, uy1)
  contoursR = imutils.grab_contours(cv2.findContours(maskR, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE))
  contoursY = imutils.grab_contours(cv2.findContours(maskY, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE))
  contours = contoursR + contoursY
  centers = []
  for i in range(len(contours)):
    x = contours[i][:, :, 0]
    y = contours[i][:, :, 1]
    xMin = x.min()
    xMax = x.max()
    yMin = y.min()
    yMax = y.max()
    centers.append(((xMin +xMax)/2,(yMin +yMax)/2,xMin,xMax,yMax,yMax))
  masks = cv2.bitwise_or(maskR, maskY)
  return contours, centers, masks

def segSingle(imgLabelled, imgGray, outputPaths, counter):
  global windowSize
  global stepSize

  contours, centers, masks = getLabelSegmentation(imgLabelled)

  bubble_threshold = 0.7
  trainingValid, trainingInvalid, labelledValid, labelledInvalid = outputPaths
  rows, cols = imgGray.shape
  for r in range(imgGray.shape[0]//stepSize - 1):
    for c in range(imgGray.shape[1]//stepSize - 1):

      left = c*stepSize
      bottom = r*stepSize
      segGray = imgGray[bottom:bottom + windowSize, left:left + windowSize]
      segLabelled = masks[bottom:bottom + windowSize, left:left + windowSize]
      
      text = 1
      for index in range(len(centers)):
        if not (left < centers[index][0] < left + windowSize and bottom < centers[index][1] < bottom + windowSize):
          continue
        
        edgeInBounds = np.bitwise_and(np.bitwise_and(contours[index][:, :, 0] >= left, contours[index][:, :, 0] <= left + windowSize),
        np.bitwise_and(contours[index][:, :, 1] >= bottom, contours[index][:, :, 1] <= bottom + windowSize))
        if np.mean(edgeInBounds) >= bubble_threshold:
          text = 0
          break
      #print(os.path.join(outputPaths[text], str(counter[text])+".jpg"))
      cv2.imwrite(os.path.join(outputPaths[text], str(counter[text])+".jpg"), segGray)
      cv2.imwrite(os.path.join(outputPaths[2+text], str(counter[text])+".jpg"), segLabelled)
      counter[text] += 1
  return counter

def segFunc(imgLabelled, imgGray, outputPaths, counter):
  global stepSize
  imgG = cv2.imread(imgGray, cv2.IMREAD_GRAYSCALE)
  imgL = cv2.imread(imgLabelled, cv2.IMREAD_COLOR)
  dim = min(imgG.shape)//windowSize
  maxDivs = 1
  while dim > 2:
    dim //= 2
    maxDivs += 1
  
  for angle in np.arange(0, 360, 45):
    dGray = imutils.rotate_bound(imgG, angle)
    dLabelled = imutils.rotate_bound(imgL, angle)
    rows, cols = dGray.shape
    for stretch in [1.0, 1.1, 1.2]:
      for i in range(maxDivs):
        dim = (int(rows * stretch) // (2**i), cols // (2**i))
        dGrayScaled = cv2.resize(dGray, dim, interpolation = cv2.INTER_AREA)
        dLabelledScaled = cv2.resize(dLabelled, dim, interpolation = cv2.INTER_AREA)

        bufferX = (stepSize - (cols % stepSize)) % stepSize
        bufferY = (stepSize - (rows % stepSize)) % stepSize

        dGrayScaled = cv2.copyMakeBorder(dGrayScaled, bufferY//2, bufferY - (bufferY//2), bufferX//2, bufferX - (bufferX//2), cv2.BORDER_REPLICATE)
        dLabelledScaled = cv2.copyMakeBorder(dLabelledScaled, bufferY//2, bufferY - (bufferY//2), bufferX//2, bufferX - (bufferX//2), cv2.BORDER_REPLICATE)
        #print(dGrayScaled.shape, dLabelledScaled.shape)

        dGrayFlipped = cv2.flip(dGrayScaled, 1)
        dLabelledFlipped = cv2.flip(dLabelledScaled, 1)
        counter = segSingle(dLabelledScaled, dGrayScaled, outputPaths, counter)
        counter = segSingle(dLabelledFlipped, dGrayFlipped, outputPaths, counter)
  return counter


counter = [0, 0]
if True:
#try:
  for manga in os.listdir(pathLabelled):
    mangaLabelledPath = os.path.join(pathLabelled, manga)
    mangaGrayPath = os.path.join(pathGray, manga)
    for chapter in os.listdir(mangaLabelledPath):
      chapterLabelledPath = os.path.join(mangaLabelledPath, chapter)
      chapterGrayPath = os.path.join(mangaGrayPath, chapter)
      for image in os.listdir(chapterLabelledPath):
        print("Running", manga, chapter, image, counter)
        imageLabelled = os.path.join(chapterLabelledPath, image)
        imageGray = os.path.join(chapterGrayPath, image)
        #print(imageLabelled, imageGray)
        counter = segFunc(imageLabelled, imageGray, pathOutput, counter)
        #raise Exception
        print("Finished", manga, chapter, image, counter)
#except Exception as e:
#  print(e)
print(counter)

