import os
import numpy as np
import cv2
import imutils
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

path = os.getcwd()
pathConvo = os.path.join(path, "OCRCONVO_DATA")
pathGray = os.path.join(pathConvo,"Grayscale")
pathMask = os.path.join(pathConvo,"Mask")
pathText = os.path.join(pathConvo, "OCR")
pathStorage = os.path.join(path, "Models", "OCR")

windowSize = 256
stepSize = windowSize // 2

epochs = 10
validation_split = 0.25
batch_size= 64

def normalize(image):
    image = np.expand_dims(np.asarray(image), axis=2)
    image = np.array(image)/255
    return image

def segSingle(imgMask, imgGray, imgSeg):
    global windowSize
    global stepSize

    X1 = []
    X2 = []
    y = []
    rows, cols = imgGray.shape
    for r in range(imgGray.shape[0]//stepSize - 1):
        for c in range(imgGray.shape[1]//stepSize - 1):
            left = c*stepSize
            bottom = r*stepSize
            segGray = imgGray[bottom:bottom + windowSize, left:left + windowSize]
            segMask = imgMask[bottom:bottom + windowSize, left:left + windowSize]
            segText = imgSeg[bottom:bottom + windowSize, left:left + windowSize]
            X1.append(normalize(segGray))
            X2.append(normalize(segText))
            y.append(normalize(segMask))
    return (X1, X2, y)
    

def segFunc(imgMask, imgGray, imgText):
    global stepSize

    X1full = []
    X2full = []
    yfull = []
    
    dim = min(imgGray.shape)//windowSize
    maxDivs = 1
    while dim > 2:
        dim //= 2
        maxDivs += 1

    for i in range(maxDivs):
        rows, cols = imgGray.shape
        dim = (rows // (2**i), cols // (2**i))
        dGrayScaled = cv2.resize(imgGray, dim, interpolation = cv2.INTER_AREA)
        dMaskScaled = cv2.resize(imgMask, dim, interpolation = cv2.INTER_AREA)
        dTextScaled = cv2.resize(imgText, dim, interpolation = cv2.INTER_AREA)
 
        bufferX = (stepSize - (cols % stepSize)) % stepSize
        bufferY = (stepSize - (rows % stepSize)) % stepSize

        dGray = cv2.copyMakeBorder(dGrayScaled, bufferY//2, bufferY - (bufferY//2), bufferX//2, bufferX - (bufferX//2), cv2.BORDER_REPLICATE)
        dMask = cv2.copyMakeBorder(dMaskScaled, bufferY//2, bufferY - (bufferY//2), bufferX//2, bufferX - (bufferX//2), cv2.BORDER_REPLICATE)
        dText = cv2.copyMakeBorder(dTextScaled, bufferY//2, bufferY - (bufferY//2), bufferX//2, bufferX - (bufferX//2), cv2.BORDER_REPLICATE)
        
        X1, X2, y = segSingle(dMask, dGray, dText)
        X1full.append(X1)
        X2full.append(X2)
        yfull.append(y)
    X1full = np.concatenate(X1full, axis=0)
    X2full = np.concatenate(X2full, axis=0)
    yfull = np.concatenate(yfull, axis=0)
        
    return (X1full, X2full, yfull)

pages = []
masks = []
textSegs = []

counter = 0
#try:
for manga in os.listdir(pathMask):
    mangaMaskPath = os.path.join(pathMask, manga)
    mangaGrayPath = os.path.join(pathGray, manga)
    mangaTextPath = os.path.join(pathText, manga)
    for chapter in os.listdir(mangaMaskPath):
        chapterMaskPath = os.path.join(mangaMaskPath, chapter)
        chapterGrayPath = os.path.join(mangaGrayPath, chapter)
        chapterTextPath = os.path.join(mangaTextPath, chapter)
        for image in os.listdir(chapterMaskPath):
            imageMask = os.path.join(chapterMaskPath, image)
            imageGray = os.path.join(chapterGrayPath, image)
            imageText = os.path.join(chapterTextPath, image)

            imgG = cv2.imread(imageGray, cv2.IMREAD_GRAYSCALE)
            imgL = cv2.imread(imageMask, cv2.IMREAD_GRAYSCALE)
            imgT = cv2.imread(imageText, cv2.IMREAD_GRAYSCALE)
            pages.append(imgG)
            masks.append(imgL)
            textSegs.append(imgT)

        #raise Exception
#except Exception as e:
  #print(e)

pages = np.array(pages)
masks = np.array(masks)
textSegs = np.array(textSegs)
print("Loaded %d pages" % pages.shape[0])


# define model
input_img = keras.Input(shape=(256, 256, 1))
input_mask = keras.Input(shape=(256, 256, 1))

# encoder branch 1
x = layers.Conv2D(16, (3, 3), activation='relu', padding='same', data_format="channels_last")(input_img)
x = layers.MaxPooling2D((2, 2), padding='same')(x)
x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = layers.MaxPooling2D((2, 2), padding='same')(x)
x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = layers.MaxPooling2D((2, 2), padding='same')(x)
xEncoded = keras.Model(input_img, x)

# encoder branch 2
y = layers.Conv2D(16, (3, 3), activation='relu', padding='same', data_format="channels_last")(input_mask)
y = layers.MaxPooling2D((2, 2), padding='same')(y)
y = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(y)
y = layers.MaxPooling2D((2, 2), padding='same')(y)
y = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(y)
y = layers.MaxPooling2D((2, 2), padding='same')(y)
yEncoded = keras.Model(input_mask, y)

# combine output of two branches
combined = layers.Concatenate()([xEncoded.output, yEncoded.output])

# decoder
z = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(combined)
z = layers.UpSampling2D((2, 2))(z)
z = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(z)
z = layers.UpSampling2D((2, 2))(z)
z = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(z)
z = layers.UpSampling2D((2, 2))(z)
decoded = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(z)

model = keras.Model([xEncoded.input, yEncoded.input], decoded)
#model.summary()

model.compile(optimizer='adam',
              loss="binary_crossentropy",
              metrics=['accuracy'])

datagen = ImageDataGenerator(rotation_range=360,
                             shear_range=45.0,
                             zoom_range=[0.5, 1.0],
                             horizontal_flip=True)


chunks = 15
chunksize = len(pages)//chunks
for i in range(chunks):
    print("chunk %d" % i)
    X1full = []
    X2full = []
    yfull = []

    for j in range(i*chunksize, (i+1)*chunksize):
        if j >= len(pages):
            break
        X1, X2, y = segFunc(masks[j], pages[j], textSegs[j])

        X1full.append(X1)
        X2full.append(X2)
        yfull.append(y)
    
    X1full = np.concatenate(X1full, axis=0)
    X2full = np.concatenate(X2full, axis=0)
    yfull = np.concatenate(yfull, axis=0)
    print("Loaded Data")
    
    perm = np.random.permutation(X1full.shape[0])
    trainIndices = 1 - int(X1full.shape[0]*validation_split)
    X1tr = X1full[perm[:trainIndices]]
    X2tr = X2full[perm[:trainIndices]]
    ytr = yfull[perm[:trainIndices]]

    X1ts = X1full[perm[trainIndices:]]
    X2ts = X2full[perm[trainIndices:]]
    yts = yfull[perm[trainIndices:]]
    
    trainGen = datagen.flow([X1tr, X2tr], ytr, batch_size=batch_size)
    valGen = datagen.flow([X1ts, X2ts], yts, batch_size=batch_size)

    
    model.fit(trainGen,
              steps_per_epoch=len(X1tr) / batch_size,
              epochs=epochs,
              validation_data=valGen)


    model.save(os.path.join(pathStorage, str(i)+".h5"))
