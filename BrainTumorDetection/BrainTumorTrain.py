import cv2
import os
from PIL import Image
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import keras
from keras.utils import normalize
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
import shutil

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

class BrainTumorTraining:
    def __init__(self, noTumorDir, yesTumorDir, size):
        self.size = size 
        self.noTumorDir = noTumorDir
        self.yesTumorDir = yesTumorDir
        self.noTumor = os.listdir(noTumorDir)
        self.yesTumor = os.listdir(yesTumorDir)
        self.dataSet = None
        self.label = None
        
    def ResizeAndConvertData(self):
        dataSet = []; label = []
        # Might want to make it into one func
        for imageName in self.noTumor:
            if (imageName.split(".")[1] == "jpg"):
                image = cv2.imread(self.noTumorDir + f"/{imageName}")
                image = Image.fromarray(image, "RGB")  # Converting colour model to RGB
                image = image.resize((self.size, self.size))
                dataSet.append(np.array(image)), label.append(False) # Makeing array of images with no label
                
        for imageName in self.yesTumor:
            if (imageName.split(".")[1] == "jpg"):
                image = cv2.imread(self.yesTumorDir + f"/{imageName}")
                image = Image.fromarray(image, "RGB")  # Converting colour model to RGB
                image = image.resize((self.size, self.size))   # Resizing image
                dataSet.append(np.array(image)), label.append(True) # Makeing array of images with yes label 
        
        self.dataSet = np.array(dataSet)
        self.label = np.array(label)
        xTrain, xTest, yTrain, yTest = train_test_split(self.dataSet, self.label, train_size= 0.67, random_state= 0) # Splits the dataset into a training and test splits
        
        # Normalizing (scaling input data) train/test data 
        xTrain = normalize(xTrain, axis= 1)
        xTest = normalize(xTest, axis= 1)
        
        return xTrain, xTest, yTrain, yTest
    def BuildingModel(self,epochsAmount):
        xTrain, xTest, yTrain, yTest = self.ResizeAndConvertData()
        model = Sequential()
        
        model.add(Conv2D(32, (3,3), input_shape= (self.size,self.size, 3))) # Conv2 (convultion layer) is used to extract features from image (convultion matrix of size 3x3)
        model.add(Activation("relu")) # Layer that activates input data
        model.add(MaxPooling2D(pool_size= (2,2))) # Layer Scales data by chosing a maximal value of sub-matrix
        
        model.add(Conv2D(32, (3,3), kernel_initializer="he_uniform")) # Kenral draws samples from uniform distribution
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size= (2,2))) 
        
        model.add(Conv2D(64, (3,3), kernel_initializer="he_uniform")) # Kenral draws samples from uniform distribution
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size= (2,2))) 
        
        model.add(Flatten()) # Layer that flattens data into single vector
        model.add(Dense(64))
        model.add(Activation("relu"))
        model.add(Dropout(0.5)) # Layer that turns off some neurons to avoid over-training
        
        model.add(Dense(1)) # Output neuron binary only yes or no
        model.add(Activation("sigmoid")) # Our problem is Binary problem so we are using sigmoid function to be able to comapre outputs on interval (0,1) 
        
        model.compile(loss = "binary_crossentropy", optimizer= "adam", metrics= ["accuracy"]) #accuracy metric will help us measure the performance of our model
        model.fit(xTrain, yTrain, batch_size=16, verbose=True, epochs = epochsAmount, validation_data=(xTest, yTest), shuffle=False) # Fitting data into our model
        model.save("BrainTumor" + str(epochsAmount) + ".h5")
        shutil.move("BrainTumor" + str(epochsAmount) + ".h5", "BrainTumorDetection")
                
        

test = BrainTumorTraining(noTumorDir = "BrainTumorDetection/no", yesTumorDir= "BrainTumorDetection/yes", size = 64)
test.ResizeAndConvertData()
test.BuildingModel(epochsAmount= 10)