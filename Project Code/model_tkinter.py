import pickle
from keras.models import load_model
import numpy as np

def stacked_prediction(members, model, inputX):
    # create dataset using ensemble
    stackedX = stacked_dataset(members, inputX)
    # make a prediction
    yhat = model.predict(stackedX)
    return yhat

def stacked_dataset(members, inputX):
    stackX = Nones
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

def model_net(input_image):
	vgg_model = load_model('30epochs.h5')
	ex_model = load_model('traindata_mini_XCEPTION.61-0.63.hdf5')
	cnn_model=load_model('cnn.h5')
	resnet_model=load_model('ResNet-50.h5')
	inception_model=load_model('Inception-v3.h5')
	members = [vgg_model,ex_model,cnn_model,resnet_model,inception_model]
	model = pickle.load(open('ensembleone.sav','rb'))
	
	input_image=input_image.reshape(1,48,48,1)
	predicted_output = stacked_prediction(members, model, input_image)
	return predicted_output[0]