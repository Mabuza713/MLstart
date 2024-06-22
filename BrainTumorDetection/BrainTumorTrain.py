import cv2
import os
from PIL import Image
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.utils import normalize
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import  Activation, Dropout, Flatten, Dense

class BrainTumorTraining:
    def __init__(self, noTumorDir, yesTumorDir):
        self.noTumorDir = noTumorDir
        self.yesTumorDir = yesTumorDir
        self.noTumor = os.listdir(noTumorDir)
        self.yesTumor = os.listdir(yesTumorDir)
        self.dataSet = []
        self.label = []  
        
    def ResizeAndConvertData(self):
        # Might want to make it into one func
        for imageName in self.noTumor:
            if (imageName.split(".")[1] == "jpg"):
                image = cv2.imread(self.noTumorDir + f"/{imageName}")
                image = Image.fromarray(image, "RGB")  # Converting colour model to RGB
                image = image.resize((64, 64))
                self.dataSet.append(np.array(image)), self.label.append(False) # Makeing array of images with no label
                
        for imageName in self.yesTumor:
            if (imageName.split(".")[1] == "jpg"):
                image = cv2.imread(self.yesTumorDir + f"/{imageName}")
                image = Image.fromarray(image, "RGB")  # Converting colour model to RGB
                image = image.resize((64, 64))   # Resizing image
                self.dataSet.append(np.array(image)), self.label.append(True) # Makeing array of images with yes label 
        
        dataSet = np.array(dataSet)
        label = np.array(label)
        xTrain, xTest, yTrain, yTest = train_test_split(dataSet, label, train_size= 0.67, random_state= 0) # Splits the dataset into a training and test splits
        
        # Normalizing (scaling input data) train/test data 
        xTrain = normalize(xTrain, axis= 1)
        xTest = normalize(xTest, axis= 1)
        
    
    def BuildingModel(self):
        model = Sequential()
        
        
                
        

test = BrainTumorTraining(noTumorDir = "BrainTumorDetection/no", yesTumorDir= "BrainTumorDetection/yes")
test.ResizeAndConvertData()