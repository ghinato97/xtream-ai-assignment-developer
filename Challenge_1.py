# Ghinato Elios 18/06/2025    
# Challenge 1

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import PolynomialFeatures


class AutomatedPipeline():
    def __init__(self,DataSetPath):
        self.DataSetPath=DataSetPath

    def LoadDataset(self):
        DataSet=pd.read_csv(self.DataSetPath)

        #Remove row where na appears
        DataSetCleaned=DataSet.dropna()
        DataSetCleaned = DataSetCleaned[(DataSetCleaned.x * DataSetCleaned.y * DataSetCleaned.z != 0) & (DataSetCleaned.price > 0)]
        DataSetCleaned = DataSetCleaned.drop(columns=['depth', 'table', 'y', 'z'])
        self.DataSet=DataSetCleaned
        print(self.DataSet)
        
    def TrainTestSplit(self,Dummy=True):
        if Dummy:
            self.DataSet = pd.get_dummies(self.DataSet, columns=['cut', 'color', 'clarity'], drop_first=True)
        x=self.DataSet.drop(columns='price')
        print(x)
        y=self.DataSet.price
        print(y)
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    def LinearRegression(self):
        self.TrainTestSplit(Dummy=True)
        y_train_log = np.log(self.y_train)
        print(y_train_log)
        reg = LinearRegression()
        reg.fit(self.x_train, y_train_log)
        pred_log= reg.predict(self.x_test)
        pred= np.exp(pred_log)

        MAE=round(mean_absolute_error(self.y_test, pred), 2)
        R2_score=round(r2_score(self.y_test, pred), 4)
        print(MAE)
        print(R2_score)

    
    def PolinomialRegression(self):
        y_train_log = np.log(self.y_train)
        # Create polynomial features with degree 3
        polynomial_features = PolynomialFeatures(degree=3)
        x_poly = polynomial_features.fit_transform(self.x_train)
        poliReg = LinearRegression()
        poliReg.fit(x_poly,y_train_log)
        poly_pred_log = model.predict(self.x_test)
        poly_pred=np.exp(poly_pred_log)

        MAE=round(mean_absolute_error(self.y_test, pred), 2)
        R2_score=round(r2_score(self.y_test, pred), 4)




        
if __name__ == '__main__':
    Pipeline=AutomatedPipeline("/home/elios/Desktop/xteam_git/xtream-ai-assignment-developer/data/diamonds.csv")
    Pipeline.LoadDataset()
    Pipeline.LinearRegression()
