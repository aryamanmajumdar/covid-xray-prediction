import numpy as np
from scipy import special
import os
from PIL import Image
import cv2
import csv

import tensorflow as tf
import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense
from keras.layers import InputLayer
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import *
from keras.applications.vgg16 import VGG16
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
import itertools
import matplotlib.pyplot as plt


#-----------------------------------------------------------------------------
#Read from CSV file and prepare the training data
#-----------------------------------------------------------------------------

#Read the CSV data into a list
rows = []
f = open(r"C:\Users\aryam\Documents\covid-19-data\Covid19-dataset\train\all_data-rescaled.csv", 'r')
reader = csv.reader(f)
for row in reader:
    rows.append(row)
    
#Prepare the data further. This includes:
#Making a list of lists for the training set, to make it easier to
#reshape later;
#Making a list of labels.
training_set = []
labels = []
for r in rows:
    data_point = []
    
    str_index = 0   
    for strin in r:
        if(str_index == 0):
            labels.append(int(strin))
        else:
            data_point.append(int(strin))
        str_index += 1
        
    str_index = 0
    training_set.append(data_point)
    

#Reshape each entry in the training set into a 224x224x3 tensor, as required
#by the input layer.
#The full training set can be thought of as a 224x224x3x251 tensor,
#if you wish.
#Finally, convert it into a numpy array.
training_set_final = []
for entry in training_set:
    arr = np.asarray(entry)
    na = arr.reshape(224,224,3)
    training_set_final.append(na)

training_set_final = np.asarray(training_set_final)

#We need to change the representation of the label data a little bit.
labels_final = []
for l in labels:
    if(l == 0):
        labels_final.append([1,0,0])
    elif(l == 1):
        labels_final.append([0,1,0])
    else:
        labels_final.append([0,0,1])
        
#Finally, convert labels into a numpy array
labels_final = np.asarray(labels_final)



#----------------------------------------------------------------------------
#Read from CSV file and prepare the testing data
#----------------------------------------------------------------------------

#Read the CSV data into a list
rows = []
f = open(r"C:\Users\aryam\Documents\covid-19-data\Covid19-dataset\test\all_data-rescaled.csv", 'r')
reader = csv.reader(f)
for row in reader:
    rows.append(row)
    
#Prepare the data further. This includes:
#Making a list of lists for the training set, to make it easier to
#reshape later;
#Making a list of labels.
testing_set = []
test_labels = []
for r in rows:
    data_point = []
    
    str_index = 0   
    for strin in r:
        if(str_index == 0):
            test_labels.append(int(strin))
        else:
            data_point.append(int(strin))
        str_index += 1
        
    str_index = 0
    testing_set.append(data_point)
    

#Reshape each entry in the training set into a 224x224x3 tensor, as required
#by the input layer.
#The full training set can be thought of as a 224x224x3x251 tensor,
#if you wish.
#Finally, convert it into a numpy array.
testing_set_final = []
for entry in testing_set:
    arr = np.asarray(entry)
    na = arr.reshape(224,224,3)
    testing_set_final.append(na)

testing_set_final = np.asarray(testing_set_final)

#We need to change the representation of the label data a little bit.
test_labels_final = []
for l in test_labels:
    if(l == 0):
        test_labels_final.append([1,0,0])
    elif(l == 1):
        test_labels_final.append([0,1,0])
    else:
        test_labels_final.append([0,0,1])
        
#Finally, convert labels into a numpy array
test_labels_final = np.asarray(test_labels_final)



#--------------------------------------------------------------
#Prepare the model
#--------------------------------------------------------------

#We're using VGG16, fine-tuned
vgg16_mod = VGG16()

#Create the input layer
input_tensor = InputLayer(input_shape=(224,224,3))

#The model will be of type Sequential
model = Sequential()

#Add the newly created input tensor to the model
model.add(input_tensor)

#Copy all the layers into the sequential model except for last layer
for layer in vgg16_mod.layers[1:-1]:
    model.add(layer)
    
#Print summary of the model for debugging
model.summary()

#Make the layers not trainable
for layer in model.layers:
    layer.trainable = False
    
#Add a final dense layer
model.add(Dense(3, activation='softmax'))

#Print summary for debugging
model.summary()


#Compile the model
model.compile(Adam(learning_rate=0.0005), loss="categorical_crossentropy", metrics=['accuracy'])
        
#Run it
model.fit(training_set_final, labels_final, batch_size=10, epochs=5)

preds = model.predict(testing_set_final, batch_size=None, verbose=0, steps=None, callbacks=None, max_queue_size=10,
    workers=1, use_multiprocessing=False)


pred_ints = []
for p in preds:
    pred_ints.append(np.argmax(p))
    
cm = confusion_matrix(test_labels, pred_ints)
    
        
    
