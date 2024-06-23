#Elios Ghinato 18/06/2024
#Challenge 2

import os
import pandas as pd
from sklearn.linear_model import LinearRegression,LassoCV
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_absolute_error
import xgboost
import numpy as np
import optuna
import time
from CommonFunction import *




class ModelRegression():
    def __init__(self,DiamondData="",PerformancePath=""):

        if DiamondData=="":
            #the  Dataset is supposed to be inside a folder called "Data" located in the same folder of the script
            generalDirectory = os.path.dirname(os.path.abspath(__file__))
            DataSetPath=os.path.join(generalDirectory,"data/diamonds.csv")
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

       
        self.performanceDict=dict.fromkeys(["DATE","NAME", "MAE","R2_SCORE"])



    def LinearRegression(self):
        DatasetLoaded = LoadDataset_(self.DataSetOriginal,CollinearityFilter=True)
        x_train,x_test,y_train,y_test = TrainTestSplit_(DatasetLoaded,Dummy=True)
        y_train_log = np.log(y_train)
        reg = LinearRegression()
        reg.fit(x_train, y_train_log)
        pred_log= reg.predict(x_test)
        pred= np.exp(pred_log)
        try:
            MAE=round(mean_absolute_error(y_test, pred), 2)
            R2_score=round(r2_score(y_test, pred), 4)
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



    def GradientBoosting(self,OptunaHyperTuning=False):

        DatasetLoaded = LoadDataset_(self.DataSetOriginal,CollinearityFilter=False,CategoricalFileter=True)
        self.x_train_xbg,x_test_xbg,self.y_train_xbg,y_test_xbg = TrainTestSplit_(DatasetLoaded,Dummy=False,StandarScaler=False)
        print(self.x_train_xbg)

        if OptunaHyperTuning:
            study = optuna.create_study(direction='minimize', study_name='Diamonds XGBoost')
            study.optimize(self.objective, n_trials=100)
            xgb_opt = xgboost.XGBRegressor(**study.best_params, enable_categorical=True, random_state=42)
            xgb_opt.fit(self.x_train_xbg, self.y_train_xbg)
            xgb_opt_pred = xgb_opt.predict(x_test_xbg)
            try:
                MAE = round(mean_absolute_error(y_test_xbg, xgb_opt_pred), 2)
                R2_score = round(r2_score(y_test_xbg, xgb_opt_pred), 4)
                self.performanceDict["NAME"] = "GradientBoosting_HyperTuning"
                self.performanceDict["DATE"] = time.ctime(time.time())
                self.performanceDict["MAE"] = MAE
                self.performanceDict["R2_SCORE"] = R2_score        
                PerformanceFile_(self.ModelPerformanceFilePath,self.performanceDict)
                SaveModel(xgb_opt,self.performanceDict["NAME"])
            except Exception as e: 
                print(e)

            
        else:
            xgb = xgboost.XGBRegressor(enable_categorical=True, random_state=42)
            xgb.fit(self.x_train_xbg, self.y_train_xbg)
            print(self.x_train_xbg)
            xgb_pred = xgb.predict(x_test_xbg)
            try:
                MAE = round(mean_absolute_error(y_test_xbg, xgb_pred), 2)
                R2_score = round(r2_score(y_test_xbg, xgb_pred), 4)
                self.performanceDict["NAME"] = "GradientBoosting"
                self.performanceDict["DATE"]= time.ctime(time.time())
                self.performanceDict["MAE"]= MAE
                self.performanceDict["R2_SCORE"]= R2_score        
                PerformanceFile_(self.ModelPerformanceFilePath,self.performanceDict)
                SaveModel(xgb,self.performanceDict["NAME"])
            except Exception as e: 
                print(e)
    


    def objective(self,trial: optuna.trial.Trial) -> float:
        # Define hyperparameters to tune
        param = {
            'lambda': trial.suggest_float('lambda', 1e-8, 1.0, log=True),
            'alpha': trial.suggest_float('alpha', 1e-8, 1.0, log=True),
            'colsample_bytree': trial.suggest_categorical('colsample_bytree', [0.3, 0.4, 0.5, 0.7]),
            'subsample': trial.suggest_categorical('subsample', [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
            'learning_rate': trial.suggest_float('learning_rate', 1e-8, 1.0, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'max_depth': trial.suggest_int('max_depth', 3, 9),
            'random_state': 42,
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'enable_categorical': True
        }

        # Split the training data into training and validation sets
        x_train, x_val, y_train, y_val = train_test_split(self.x_train_xbg, self.y_train_xbg, test_size=0.2, random_state=42)

        # Train the model
        model = xgboost.XGBRegressor(**param)
        model.fit(x_train, y_train)

        # Make predictions
        preds = model.predict(x_val)

        # Calculate MAE
        mae = mean_absolute_error(y_val, preds)

        return mae



    
if __name__ == '__main__':
    Pipeline=ModelRegression()
    #Pipeline.LinearRegression()
    Pipeline.GradientBoosting()
    #Pipeline.PolinomialRegression()
    #Pipeline.LassoCVRegression()

