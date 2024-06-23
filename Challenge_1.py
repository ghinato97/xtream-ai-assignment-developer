# Ghinato Elios 18/06/2025    
# Challenge 1

import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression,LassoCV
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import PolynomialFeatures
import pickle
import time 
from CommonFunction import *

class ModelRegression():
    def __init__(self,DiamondData="",PerformancePath=""):

        if DiamondData == "":
            #the  Dataset is supposed to be inside a folder called "Data" located in the same folder of the script
            generalDirectory = os.path.dirname(os.path.abspath(__file__))
            DataSetPath = os.path.join(generalDirectory,"data/diamonds.csv")
        else:
            DataSetPath=DiamondData

        #load and remove row where na appears
        DataSet = pd.read_csv(DataSetPath)
        DataSet = DataSet.dropna()
        #remove nonsense data 
        self.DataSetOriginal = DataSet[(DataSet.x * DataSet.y * DataSet.z != 0) & (DataSet.price > 0)]

    
        if PerformancePath == "":
            self.ModelPerformanceFilePath = os.path.join(generalDirectory,"data/ModelPerformance.csv")
        else:
            self.ModelPerformanceFilePath = PerformancePath

        self.performanceDict = dict.fromkeys(["DATE","NAME", "MAE","R2_SCORE"])

    
    def LinearRegression(self):
        DatasetLoaded = LoadDataset_(self.DataSetOriginal,CollinearityFilter=True)
        x_train, x_test, y_train, y_test = TrainTestSplit_(DatasetLoaded,Dummy=True)

        y_train_log = np.log(y_train)
        reg = LinearRegression()
        reg.fit(x_train, y_train_log)
        pred_log = reg.predict(x_test)
        pred = np.exp(pred_log)

        try:

            MAE = round(mean_absolute_error(y_test, pred), 2)
            R2_score = round(r2_score(y_test, pred), 4)
            self.performanceDict["NAME"] = "LinearRegression"
            self.performanceDict["DATE"] = time.ctime(time.time())
            self.performanceDict["MAE"] = MAE
            self.performanceDict["R2_SCORE"] = R2_score        
            PerformanceFile_(self.ModelPerformanceFilePath,self.performanceDict)
        except Exception as e: 
            print(e)




    def PolinomialRegression(self):
        DatasetLoaded=LoadDataset_(self.DataSetOriginal,CollinearityFilter=True)
        x_train,x_test,y_train,y_test = TrainTestSplit_(DatasetLoaded,Dummy=True)

        y_train_log = np.log(y_train)
        # Create polynomial features with degree 3 for Test and Train input dataset
        polynomial_features = PolynomialFeatures(degree=3)
        x_train_poly = polynomial_features.fit_transform(x_train)
        x_test_poly = polynomial_features.fit_transform(x_test)
        poliReg = LinearRegression()
        poliReg.fit(x_train_poly,y_train_log)
        poly_pred_log = poliReg.predict(x_test_poly)
        poly_pred = np.exp(poly_pred_log)

        try:
            MAE=round(mean_absolute_error(y_test, poly_pred), 2)
            R2_score = round(r2_score(y_test, poly_pred), 4)
            self.performanceDict["NAME"] = "PolinomialRegression"
            self.performanceDict["DATE"] = time.ctime(time.time())
            self.performanceDict["MAE"] = MAE
            self.performanceDict["R2_SCORE"] = R2_score        
            PerformanceFile_(self.ModelPerformanceFilePath,self.performanceDict)
        except Exception as e: 
            print(e)




    def LassoCVRegression(self):
        DatasetLoaded=LoadDataset_(self.DataSetOriginal,CollinearityFilter=True)
        x_train,x_test,y_train,y_test = TrainTestSplit_(DatasetLoaded,Dummy=True,StandarScaler=True)
        lasso_model = LassoCV()
        lasso_model.fit(x_train, y_train)
        lassoPred = lasso_model.predict(x_test)
        try:
            MAE=round(mean_absolute_error(y_test, lassoPred), 2)
            R2_score=round(r2_score(y_test, lassoPred), 4)
            self.performanceDict["NAME"] = "LassoCVRegression"
            self.performanceDict["DATE"] = time.ctime(time.time())
            self.performanceDict["MAE"] = MAE
            self.performanceDict["R2_SCORE"] = R2_score
            PerformanceFile_(self.ModelPerformanceFilePath,self.performanceDict)
        except Exception as e: 
            print(e)







if __name__=='__main__':        
    Pipeline=ModelRegression()
    Pipeline.LinearRegression()
    Pipeline.PolinomialRegression()
    Pipeline.LassoCVRegression()
