from scipy.io import loadmat
import pandas as pd
import numpy as np
from random import shuffle
import os
import cv2

def loaddataset(filename):
  data = pd.read_csv(filename)
  pixels = data['pixels'].tolist()
  width, height = 48, 48
  faces = []
  for pixel_sequence in pixels:
      face = [int(pixel) for pixel in pixel_sequence.split(' ')]
      face = np.asarray(face)#.reshape(width, height)
  #     face = cv2.resize(face.astype('uint8'),(48,48))
      faces.append(face.astype('float32'))
  faces = np.asarray(faces)
  faces = np.expand_dims(faces, -1)
  emotions = pd.get_dummies(data['emotion']).as_matrix()
  return faces,emotions
trainx,trainy = loaddataset('Train_Data.csv')
testx,testy=loaddataset('Test_Data.csv')
valx,valy=loaddataset('Validation_Data.csv')

trainx1=trainx.astype('float32') / 255
mean, std = np.mean(trainx1), np.std(trainx1)
train_X = np.asarray([(np.array(xi)-mean) for xi in trainx1])


testx1=testx.astype('float32') / 255
mean, std = np.mean(testx1), np.std(testx1)
test_X = np.asarray([(np.array(xi)-mean) for xi in testx1])

valx1=valx.astype('float32') / 255
mean, std = np.mean(valx1), np.std(valx1)
val_X = np.asarray([(np.array(xi)-mean) for xi in valx1])

train_X = train_X.reshape(train_X.shape[0], 48, 48)
train_X = train_X.reshape(train_X.shape[0], 48, 48, 1)
val_X = val_X.reshape(val_X.shape[0],48, 48)
val_X = val_X.reshape(val_X.shape[0],48, 48,1)
test_X = test_X.reshape(test_X.shape[0],48, 48)  
test_X = test_X.reshape(test_X.shape[0],48, 48, 1) 

train_Y=trainy
val_Y=valy
test_Y=testy

img_rows, img_cols = 48, 48
batch_size = 64

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.normalization import BatchNormalization
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import ZeroPadding2D
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=0.3,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.3,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=True,  # randomly flip images
    vertical_flip=False)  # randomly flip images

valid_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

datagen.fit(train_X)

train_generator=datagen.flow(train_X, train_Y,batch_size=batch_size)

validation_generator = valid_datagen.flow(val_X,val_Y,batch_size=batch_size)

test_generator = valid_datagen.flow(test_X,test_Y,batch_size=batch_size)

number_of_classes = 7
dimension = 48
number_of_channels = 1
model = Sequential()
model.add(Conv2D(64, (3, 3), input_shape=(48, 48 ,1), padding='same', activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(number_of_classes, activation='softmax'))

# Compile model
epochs = 100
lrate = 0.01
decay = lrate/epochs
adam = Adam(decay=decay)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])


model.fit_generator(train_generator,
                    samples_per_epoch=train_X.shape[0],
                    nb_epoch=epochs,
                    validation_data=(val_X, val_Y))

train_loss, test_acc = model.evaluate_generator(generator=train_generator,steps=batch_size)
print("Train Accuracy: "+ repr(test_acc*100) + '%')
print("Train Loss: " + repr(train_loss))

train_loss, test_acc = model.evaluate_generator(generator=validation_generator,steps=batch_size)
print("Validation Accuracy: "+ repr(test_acc*100) + '%')
print("Validation Loss: " + repr(train_loss))

train_loss, test_acc = model.evaluate_generator(generator=test_generator,steps=batch_size)
print("Testing Accuracy: "+ repr(test_acc*100) + '%')
print("Testing Loss: " + repr(train_loss))

# model.save('100epochs.h5')
