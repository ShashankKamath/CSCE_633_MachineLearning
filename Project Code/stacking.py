from keras.models import load_model
import numpy as np
from scipy.io import loadmat
import pandas as pd
from random import shuffle
import os
import cv2
from numpy import dstack
from sklearn.linear_model import LogisticRegression

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


vgg_model = load_model('100epochs.h5')
# print("VGG MODEL")
# train_loss, test_acc = vgg_model.evaluate(train_X, train_Y)
# print("Accuracy: "+ repr(test_acc*100) + '%')
# print("Train Loss: " + repr(train_loss))
# train_loss, test_acc = vgg_model.evaluate(val_X, val_Y)
# print("Accuracy: "+ repr(test_acc*100) + '%')
# print("Validation Loss: " + repr(train_loss))
# test_loss, test_acc = vgg_model.evaluate(test_X, test_Y)
# print("Accuracy: "+ repr(test_acc*100) + '%')
# print("Test Loss: " + repr(test_loss))

ex_model = load_model('traindata_mini_XCEPTION.61-0.63.hdf5')
# print("MiniException MODEL")
# train_loss, test_acc = ex_model.evaluate(train_X, train_Y)
# print("Accuracy: "+ repr(test_acc*100) + '%')
# print("Train Loss: " + repr(train_loss))
# train_loss, test_acc = ex_model.evaluate(val_X, val_Y)
# print("Accuracy: "+ repr(test_acc*100) + '%')
# print("Validation Loss: " + repr(train_loss))
# test_loss, test_acc = ex_model.evaluate(test_X, test_Y)
# print("Accuracy: "+ repr(test_acc*100) + '%')
# print("Test Loss: " + repr(test_loss))

members = [vgg_model,ex_model]

# create stacked model input dataset as outputs from the ensemble
def stacked_dataset(members, inputX):
    stackX = None
    for model in members:
        # make prediction
        yhat = model.predict(inputX, verbose=0)
        # stack predictions into [rows, members, probabilities]
        if stackX is None:
            stackX = yhat
        else:
            stackX = dstack((stackX, yhat))
    # flatten predictions to [rows, members x probabilities]
    stackX = stackX.reshape((stackX.shape[0], stackX.shape[1]*stackX.shape[2]))
    return stackX

# fit a model based on the outputs from the ensemble members
def fit_stacked_model(members, inputX, inputy):
    # create dataset using ensemble
    stackedX = stacked_dataset(members, inputX)
    # fit standalone model
    model = LogisticRegression()
    model.fit(stackedX, inputy)
    return model

# fit stacked model using the ensemble
model = fit_stacked_model(members, test_X, test_Y)

# make a prediction with the stacked model
def stacked_prediction(members, model, inputX):
    # create dataset using ensemble
    stackedX = stacked_dataset(members, inputX)
    # make a prediction
    yhat = model.predict(stackedX)
    return yhat

# evaluate model on test set
yhat = stacked_prediction(members, model, test_X)
acc = accuracy_score(test_Y, yhat)
print('Stacked Test Accuracy: %.3f' % acc)