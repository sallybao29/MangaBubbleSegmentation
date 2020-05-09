import pickle

import numpy as np
import scipy
from sklearn import linear_model
import cv2
import os

path = os.getcwd()
pathStorage = os.path.join(path, "Models", "Log")
pathSource = (os.path.join(path, "TrainingValid"), os.path.join(path, "TrainingInvalid"))

pickleOutput = os.path.join(pathStorage, str(len(os.listdir(pathStorage))) + ".p")

def readImages(basePaths, training, testing):
  validBase, invalidBase = basePaths
  lenValid = len(os.listdir(validBase))
  lenInvalid = len(os.listdir(invalidBase))
  print(lenValid, lenInvalid)
  rand_indices_valid = np.random.permutation(lenValid)[:training + testing]
  rand_indices_invalid = np.random.permutation(lenInvalid)[:training + testing]
  print("finished random")

  #assume 0.jpg exists
  imageSize = np.ravel(cv2.imread(os.path.join(validBase, "0.jpg"), cv2.IMREAD_GRAYSCALE)).shape[0]
  Xtr = np.zeros((training, imageSize))
  ytr = np.zeros(training)
  Xts = np.zeros((testing, imageSize))
  yts = np.zeros(testing)

  training //= 2
  testing //= 2

  ytr[:training] = 1
  ytr[training:] = 0

  yts[:testing] = 1
  yts[testing:] = 0

  for i in range(training):
    Xtr[i,:] = np.ravel(cv2.imread(os.path.join(validBase, str(rand_indices_valid[i]) + ".jpg"), cv2.IMREAD_GRAYSCALE))
    Xtr[i+training,:] = np.ravel(cv2.imread(os.path.join(invalidBase, str(rand_indices_invalid[i]) + ".jpg"), cv2.IMREAD_GRAYSCALE))
    if i % 1000 == 0:
      print("loaded",i,"of",training)

  for i in range(testing):
    Xts[i,:] = np.ravel(cv2.imread(os.path.join(validBase, str(rand_indices_valid[i + training]) + ".jpg"), cv2.IMREAD_GRAYSCALE))
    Xts[i+testing,:] = np.ravel(cv2.imread(os.path.join(invalidBase, str(rand_indices_invalid[i + training]) + ".jpg"), cv2.IMREAD_GRAYSCALE))
    if i % 1000 == 0:
      print("loaded",i,"of",testing)
  return Xtr,ytr,Xts,yts

"""
def readImages(basePaths, paths):
  validBase, invalidBase = basePaths
  X = []
  y = []
  for path in paths:
    validImage = cv2.imread(os.path.join(validBase, str(path)+".jpg"), cv2.IMREAD_GRAYSCALE)
    invalidImage = cv2.imread(os.path.join(invalidBase, str(path)+".jpg"), cv2.IMREAD_GRAYSCALE)
    if validImage is not None:
      X.append(np.ravel(validImage))
      y.append(1)
    if invalidImage is not None:
      X.append(np.ravel(invalidImage))
      y.append(0)
  return np.array(X), np.array(y)
"""
ntr = 20000
nts = 20000
offset = 0

#rand_indices = np.random.permutation(ntr+nts)

#rand_indices_tr = rand_indices[offset:ntr+offset]
#rand_indices_ts = rand_indices[ntr+offset:ntr+offset+nts]

#Xtr_raw, ytr = readImages(pathSource, rand_indices_tr)
#Xts_raw, yts = readImages(pathSource, rand_indices_ts)
print("Starting Read")
Xtr, ytr, Xts, yts = readImages(pathSource, ntr, nts)

Xtr/=255
Xts/=255

Xtr-=0.5
Xts-=0.5

logreg = linear_model.LogisticRegression(penalty='l2', solver='newton-cg', max_iter=500)
print("Starting fit. Waiting")
logreg.fit(Xtr, ytr)
print("Finished fit.")
yhat = logreg.predict(Xts)
acc = np.mean(yhat == yts)
print('Logistic Regression Accuracy = {0:f}'.format(acc))

with open(pickleOutput, "wb" ) as fp:
  pickle.dump(logreg, fp)
  print("Stored model in", pickleOutput)
