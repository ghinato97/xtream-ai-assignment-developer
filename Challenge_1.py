# Ghinato Elios 18/06/2025    
# Challenge 1

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,LassoCV
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import PolynomialFeatures,StandardScaler
import pickle



class AutomatedPipeline():
    def __init__(self,DataSetPath):
        self.DataSetPath=DataSetPath

    def LoadDataset(self):
        DataSet=pd.read_csv(self.DataSetPath)
      
        #Remove row where na appears
        DataSetCleaned=DataSet.dropna()
        DataSetCleaned = DataSetCleaned[(DataSetCleaned.x * DataSetCleaned.y * DataSetCleaned.z != 0) & (DataSetCleaned.price > 0)]
        DataSetCleaned = DataSetCleaned.drop(columns=['depth', 'table', 'y', 'z'])
        self.DataSetOriginal=DataSetCleaned
        
    def TrainTestSplit(self,Dummy=False,StandarScaler=False):
        if Dummy:
            DataSet = pd.get_dummies(self.DataSetOriginal, columns=['cut', 'color', 'clarity'], drop_first=True)
        else:
            DataSet=self.DataSetOriginal
        x=DataSet.drop(columns='price')
        y=DataSet.price
        if StandarScaler:
            scaler = StandardScaler()
            x_scaled = scaler.fit_transform(x)
            return (train_test_split(x_scaled, y, test_size=0.2, random_state=42))
        else:            
            return (train_test_split(x, y, test_size=0.2, random_state=42))

    def LinearRegression(self):
        x_train,x_test,y_train,y_test = self.TrainTestSplit(Dummy=True,StandarScaler=True)
        y_train_log = np.log(y_train)
        reg = LinearRegression()
        reg.fit(x_train, y_train_log)
        pred_log = reg.predict(x_test)
        pred = np.exp(pred_log)

        MAE = round(mean_absolute_error(y_test, pred), 2)
        R2_score = round(r2_score(y_test, pred), 4)
        print(MAE)
        print(R2_score)

    
    def PolinomialRegression(self):
        x_train,x_test,y_train,y_test = self.TrainTestSplit(Dummy=True,StandarScaler=False)
        y_train_log = np.log(y_train)
        # Create polynomial features with degree 3 for Test and Train input dataset
        polynomial_features = PolynomialFeatures(degree=3)
        x_train_poly = polynomial_features.fit_transform(x_train)
        x_test_poly = polynomial_features.fit_transform(x_test)
        poliReg = LinearRegression()
        poliReg.fit(x_train_poly,y_train_log)
        poly_pred_log = poliReg.predict(x_test_poly)
        poly_pred=np.exp(poly_pred_log)

        try:
            MAE=round(mean_absolute_error(y_test, poly_pred), 2)
            R2_score=round(r2_score(y_test, poly_pred), 4)
            print(MAE)
            print(R2_score)
        except Exception as e: 
            print(e)

    def LassoCVRegression(self):
        x_train,x_test,y_train,y_test = self.TrainTestSplit(Dummy=True,StandarScaler=True)
        lasso_model = LassoCV()
        lasso_model.fit(x_train, y_train)
        lassoPred=lasso_model.predict(x_test)
        try:
            MAE=round(mean_absolute_error(y_test, lassoPred), 2)
            R2_score=round(r2_score(y_test, lassoPred), 4)
            print(MAE)
            print(R2_score)
        except Exception as e: 
            print(e)







        
if __name__ == '__main__':
    Pipeline=AutomatedPipeline("/home/elios/Desktop/xteam_git/xtream-ai-assignment-developer/data/diamonds.csv")
    Pipeline.LoadDataset()
    Pipeline.LinearRegression()
    Pipeline.PolinomialRegression()
    Pipeline.LassoCVRegression()
