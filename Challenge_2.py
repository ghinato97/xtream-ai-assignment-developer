#Elios Ghinato Challenge 2
# Ghinato Elios 18/06/2025    
# Challenge 1

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
import xgboost


class AutomatedPipeline2():
    def __init__(self,DataSetPath):
        #load and remove row where na appears
        DataSet=pd.read_csv(DataSetPath)
        DataSet=DataSet.dropna()
        #remove nonsense data 
        self.DataSet = DataSet[(DataSet.x * DataSet.y * DataSet.z != 0) & (DataSet.price > 0)]



    def LoadDataset(self,CollinearityFilter=False,CategoricalFileter=False):
        DataSet=self.DataSet
        if CollinearityFilter: #For LinearRegression
            DataSetCollinearity = DataSet.drop(columns=['depth', 'table', 'y', 'z'])
            self.DataSetLoaded = DataSetCollinearity
        if CategoricalFileter: #For xgboost
            DataSet_xgb=DataSet
            print(DataSet_xgb)
            DataSet_xgb['cut'] = pd.Categorical(DataSet['cut'], categories=['Fair', 'Good', 'Very Good', 'Ideal', 'Premium'], ordered=True)
            DataSet_xgb['color'] = pd.Categorical(DataSet['color'], categories=['D', 'E', 'F', 'G', 'H', 'I', 'J'], ordered=True)
            DataSet_xgb['clarity'] = pd.Categorical(DataSet['clarity'], categories=['IF', 'VVS1', 'VVS2', 'VS1', 'VS2', 'SI1', 'SI2', 'I1'], ordered=True)
            self.DataSetLoaded = DataSet_xgb
            print(self.DataSetLoaded.info())


    def TrainTestSplit(self,Dummy=False):
        if Dummy:
            DataSet = pd.get_dummies(self.DataSetLoaded, columns=['cut', 'color', 'clarity'], drop_first=True)
        else:
            DataSet=self.DataSetLoaded
        x=DataSet.drop(columns='price')
        y=DataSet.price
        return (train_test_split(x, y, test_size=0.2, random_state=42))

    def LinearRegression(self):
        self.LoadDataset(CollinearityFilter=True,CategoricalFileter=False)
        x_train,x_test,y_train,y_test = self.TrainTestSplit(Dummy=True)
        print(x_train)
        y_train_log = np.log(y_train)
        reg = LinearRegression()
        reg.fit(x_train, y_train_log)
        pred_log= reg.predict(x_test)
        pred= np.exp(pred_log)

        MAE=round(mean_absolute_error(y_test, pred), 2)
        R2_score=round(r2_score(y_test, pred), 4)
        print(MAE)
        print(R2_score)

    
    def GradientBoosting(self):
        self.LoadDataset(CollinearityFilter=False,CategoricalFileter=True)
        x_train_xbg,x_test_xbg,y_train_xbg,y_test_xbg = self.TrainTestSplit(Dummy=False)
        print(x_train_xbg.info())
        print(x_test_xbg.info())
        xgb = xgboost.XGBRegressor(enable_categorical=True, random_state=42)
        xgb.fit(x_train_xbg, y_train_xbg)
        xgb_pred = xgb.predict(x_test_xbg)

        MAE=round(mean_absolute_error(y_test_xbg, xgb_pred), 2)
        R2_score=round(r2_score(y_test_xbg, xgb_pred), 4)
        print(MAE)
        print(R2_score)
    




        
if __name__ == '__main__':
    Pipeline=AutomatedPipeline2("/home/elios/Desktop/xteam_git/xtream-ai-assignment-developer/data/diamonds.csv")
    Pipeline.LinearRegression()
    Pipeline.GradientBoosting()

