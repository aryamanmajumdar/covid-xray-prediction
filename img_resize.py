#--------------------------------------------------------------------------------------
#Aryaman Majumdar
#
#24th May, 2021
#
#Predicting between COVID-19, viral pneumonia and other from X-Ray images using a CNN.
#Here, we prepare the data - rescaling the original JPGs/JPEGs and converting to CSV.
#--------------------------------------------------------------------------------------

"""
Created on Sat May 22 10:39:47 2021

@author: aryam
"""

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
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import *
from keras.applications.vgg16 import VGG16
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt




#Returns list of files
def createFileList(myDir):
    
    fileList = []
    
    for root, dirs, files in os.walk(myDir, topdown=False):
        for name in files:
            if (name.endswith(".jpg") or name.endswith(".jpeg") or name.endswith(".png")):
                fullName = os.path.join(root, name)
                fileList.append(fullName)
    return fileList

#Resizes the images pointed to by the argument fileList
def resizeImages(fileList, outputDir):
    
    name_int = 0
    name = ""
    
    for file in fileList:
        
        name = str(name_int) + ".jpg"
        
        #read image
        img = cv2.imread(file)

        #resize image
        res = cv2.resize(img, dsize=(224,224), interpolation=cv2.INTER_CUBIC)

        #make image grayscale
        #res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)

        cv2.imwrite(os.path.join(outputDir , name), res)
        
        name_int += 1
        

#We still have to change this method.
#The first entry in each record (row) in the CSV
#can be the target value (0, 1, 2 for covid, normal, and pneumonia)
#This is simple. Search the csv_output_directory string for the term
#'normal', 'pneumonia' or 'covid'. More specifically,
#if the string contains 'normal', the target is 1. 
#If the string contains 'pneumonia', the target is 2.
#Else the target is 0 (implying covid).
def makeCSV(fileList, csv_output_directory):
    for file in fileList:
        #read image using opencv
        img = cv2.imread(file)
    
        # get original image parameters
        width, height, channels = img.shape
        
    
# =============================================================================
#         # Make image Greyscale
#         #img_grey = img_file.convert('L')
#         
#         img_2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         img_file = Image.fromarray(img_2)
# 
#         # Save  values
#         value = np.asarray(img_file.getdata(), dtype=np.int).reshape((img_file.size[1], img_file.size[0]))
        value = img.flatten()
# =============================================================================
        

        #Open a file; make a file with the appropriate name
        #if it doesn't exist.
        f = open(csv_output_directory, 'a')
        
        #Create a CSV writer to make things easier
        writer = csv.writer(f, lineterminator='\n')
        
        if "normal" in csv_output_directory:
            f.write("1")
        elif "pneumonia" in csv_output_directory:
            f.write("2")
        else:
            f.write("0")
        
        #We have to write the first comma separately
        f.write(',')
        
        #The writerow method writes the pixel values,
        #separated by commas (because the writerow method is
        # a method of the csv.writer class)
        writer.writerow(value)
        
        #Close the writer
        f.close()
        

#Create file lists for: 
#1) covid training
#2) normal training
#3) viral pneumonia training
#4) covid testing
#5) normal training
#6) viral pneumonia training


#--------------------------------------------------------------------
#Prepare the training data.
#--------------------------------------------------------------------

#Creating file lists for resizing the training sets
fl_covid_train = createFileList(r"C:\Users\aryam\Documents\covid-19-data\Covid19-dataset\train\covid")
fl_normal_train = createFileList(r"C:\Users\aryam\Documents\covid-19-data\Covid19-dataset\train\normal")
fl_vp_train = createFileList(r"C:\Users\aryam\Documents\covid-19-data\Covid19-dataset\train\viral-pneumonia")

#Resizing training images
resizeImages(fl_covid_train, r"C:\Users\aryam\Documents\covid-19-data\Covid19-dataset\train\covid-rescaled")
resizeImages(fl_normal_train, r"C:\Users\aryam\Documents\covid-19-data\Covid19-dataset\train\normal-rescaled")
resizeImages(fl_vp_train, r"C:\Users\aryam\Documents\covid-19-data\Covid19-dataset\train\viral-pneumonia-rescaled")
 
#Creating file lists of the resized images
fl_covid_train_resized = createFileList(r"C:\Users\aryam\Documents\covid-19-data\Covid19-dataset\train\covid-rescaled")
fl_normal_train_resized = createFileList(r"C:\Users\aryam\Documents\covid-19-data\Covid19-dataset\train\normal-rescaled")
fl_vp_train_resized = createFileList(r"C:\Users\aryam\Documents\covid-19-data\Covid19-dataset\train\viral-pneumonia-rescaled")

#Make CSV files from resized images
makeCSV(fl_covid_train_resized, r"C:\Users\aryam\Documents\covid-19-data\Covid19-dataset\train\covid-rescaled-csv\covid-rescaled.csv")
makeCSV(fl_normal_train_resized, r"C:\Users\aryam\Documents\covid-19-data\Covid19-dataset\train\normal-rescaled-csv\normal-rescaled.csv")
makeCSV(fl_vp_train_resized, r"C:\Users\aryam\Documents\covid-19-data\Covid19-dataset\train\viral-pneumonia-rescaled-csv\viral-pneumonia-rescaled.csv")


#--------------------------------------------------------------------
#Prepare the testing data.
#--------------------------------------------------------------------

#Creating file lists for testing sets
fl_covid_test = createFileList(r"C:\Users\aryam\Documents\covid-19-data\Covid19-dataset\test\covid")
fl_normal_test = createFileList(r"C:\Users\aryam\Documents\covid-19-data\Covid19-dataset\test\normal")
fl_vp_test = createFileList(r"C:\Users\aryam\Documents\covid-19-data\Covid19-dataset\test\viral-pneumonia")

#Resizing testing images
resizeImages(fl_covid_test, r"C:\Users\aryam\Documents\covid-19-data\Covid19-dataset\test\covid-rescaled")
resizeImages(fl_normal_test, r"C:\Users\aryam\Documents\covid-19-data\Covid19-dataset\test\normal-rescaled")
resizeImages(fl_vp_test, r"C:\Users\aryam\Documents\covid-19-data\Covid19-dataset\test\viral-pneumonia-rescaled")

#Creating file lists of the resized images
fl_covid_test_resized = createFileList(r"C:\Users\aryam\Documents\covid-19-data\Covid19-dataset\test\covid-rescaled")
fl_normal_test_resized = createFileList(r"C:\Users\aryam\Documents\covid-19-data\Covid19-dataset\test\normal-rescaled")
fl_vp_test_resized = createFileList(r"C:\Users\aryam\Documents\covid-19-data\Covid19-dataset\test\viral-pneumonia-rescaled")

#Make CSV files from resized images
makeCSV(fl_covid_test_resized, r"C:\Users\aryam\Documents\covid-19-data\Covid19-dataset\test\covid-rescaled-csv\covid-rescaled.csv")
makeCSV(fl_normal_test_resized, r"C:\Users\aryam\Documents\covid-19-data\Covid19-dataset\test\normal-rescaled-csv\normal-rescaled.csv")
makeCSV(fl_vp_test_resized, r"C:\Users\aryam\Documents\covid-19-data\Covid19-dataset\test\viral-pneumonia-rescaled-csv\viral-pneumonia-rescaled.csv")



