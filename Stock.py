import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
import yfinance as yf
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense




class StockPredictor:
    def __init__(self, stockName):
        self.stockName = stockName



  
    def GetData(self):
        dates = [0]
        prices = []
        
        data = yf.download(self.stockName, period= "1mo")
        for index, rows in data.iterrows():
            
            prices.append(rows.array[0])
            
        dates = range(0,len(prices))
        return [prices, dates]
    
    def PredictPrices(self):
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

result =  StockPredictor("BTC-USD")
result.PredictPrices()

