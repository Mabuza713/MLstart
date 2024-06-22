import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
import yfinance as yf
import pandas as pd
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
import math

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"



class StockPredictor:
    def __init__(self, stockName):
        self.stockName = stockName



  
    def GetData(self):
        dates = [0]
        prices = []
        
        data = yf.download(self.stockName, period= "6mo")
        for index, rows in data.iterrows():
            
            prices.append(rows.array[0])
            
        dates = range(0,len(prices))
        return [prices, dates]
    
    def SupportiveVectorRegression(self):
        data = self.GetData()
        prices = data[0]; dates = data[1]
        dates = np.reshape(dates, (len(dates),1))
        
        svrLin = SVR(kernel= "linear", C=1e3)
        svrPoly = SVR(kernel= "poly", C=1e3, degree = 2)
        svrRbf = SVR(kernel = "rbf", C=1e3, gamma = 0.1)
        
        svrLin.fit(dates, prices); svrPoly.fit(dates, prices)
        svrRbf.fit(dates, prices)
        
        plt.scatter(dates, prices, color="black", label = "Data")
        plt.plot(dates, svrRbf.predict(dates), color="blue", label = "Radial Basis Function - RBF")
        plt.plot(dates, svrLin.predict(dates), color="red", label = "Linear model")
        plt.plot(dates, svrPoly.predict(dates), color="green", label = "Polynomial model")        
        
        plt.xlabel("Date")
        plt.ylabel("Prices")
        plt.title("Support Vector Regression - SVR")
        plt.legend()
        plt.show()
        
        return

    def CreateMatrixDataSet(self, dataSet, lookBack = 1):
        dataX, dataY = [dataSet[0]], []
        for i in range(1,len(dataSet) - 1):
            dataX.append(dataSet[i])
            dataY.append(dataX[-1])
        dataY.append(dataSet[-1])
        return np.array(dataX), np.array(dataY)
    
    
    def PredicStockPrice(self, trainToTestProp = 0.67):
        # Getting and preparing data
        data = self.GetData()
        dataSet = data[0]; dates = data[1]
        plt.plot(dataSet)
        
        trainSize = int(len(dataSet) * trainToTestProp)
        
        train, test = dataSet[0:trainSize], dataSet[trainSize:]
        
        # Create train and test dataSets 
        trainX, trainY = self.CreateMatrixDataSet(train)
        testX, testY = self.CreateMatrixDataSet(test)
        
        # Creating perceptron model
        model = Sequential()
        model.add(Dense(8, input_shape = (1, ), activation="relu"))
        model.add(Dense(1))
        model.compile(loss = "mean_squared_error", optimizer = "adam")
        model.fit(trainX, trainY, epochs = 400, batch_size = 3, verbose = 2)
        
        # Stats of perfomed model
        trainScore = model.evaluate(trainX, trainY, verbose = 0)
        testScore = model.evaluate(testX, testY, verbose = 0)
        print(f"train score: {math.sqrt(trainScore)}")
        print(f"test score: {math.sqrt(testScore)}")
        
        # Plotting predictions 
        trainPredict = model.predict(trainX)
        testPredict = model.predict(testX)
        trainPredict = np.transpose(trainPredict)[0]; testPredict = np.transpose(testPredict)[0]
        
        final = trainPredict.tolist()
        for value in testPredict:
            final.append(value)   
        
        
        plt.plot(final)
        
        plt.show()    

result =  StockPredictor("BTC-USD")
result.PredicStockPrice()
result.SupportiveVectorRegression()



