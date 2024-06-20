#Elios Ghinato 18/06/2024
#Challenge 2

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
import xgboost
import sklearn
import optuna




class AutomatedPipeline2():
    def __init__(self,DataSetPath):
        #load and remove row where na appears
        DataSet=pd.read_csv(DataSetPath)
        DataSet=DataSet.dropna()
        #remove nonsense data 
        self.DataSetOriginal = DataSet[(DataSet.x * DataSet.y * DataSet.z != 0) & (DataSet.price > 0)]



    def LoadDataset(self,CollinearityFilter=False,CategoricalFileter=False):
        DataSet=self.DataSetOriginal
        if CollinearityFilter: #For LinearRegression
            DataSetCollinearity = DataSet.drop(columns=['depth', 'table', 'y', 'z'])
            self.DataSetLoaded = DataSetCollinearity
        if CategoricalFileter: #For xgboost
            DataSet_xgb=DataSet
            DataSet_xgb['cut'] = pd.Categorical(DataSet['cut'], categories=['Fair', 'Good', 'Very Good', 'Ideal', 'Premium'], ordered=True)
            DataSet_xgb['color'] = pd.Categorical(DataSet['color'], categories=['D', 'E', 'F', 'G', 'H', 'I', 'J'], ordered=True)
            DataSet_xgb['clarity'] = pd.Categorical(DataSet['clarity'], categories=['IF', 'VVS1', 'VVS2', 'VS1', 'VS2', 'SI1', 'SI2', 'I1'], ordered=True)
            self.DataSetLoaded = DataSet_xgb


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
        y_train_log = np.log(y_train)
        reg = LinearRegression()
        reg.fit(x_train, y_train_log)
        pred_log= reg.predict(x_test)
        pred= np.exp(pred_log)

        MAE=round(mean_absolute_error(y_test, pred), 2)
        R2_score=round(r2_score(y_test, pred), 4)
        print(MAE)
        print(R2_score)

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







    def GradientBoosting(self,OptunaHyperTuning=False):
        self.LoadDataset(CollinearityFilter=False,CategoricalFileter=True)
        self.x_train_xbg,x_test_xbg,self.y_train_xbg,y_test_xbg = self.TrainTestSplit(Dummy=False)
        print(self.x_train_xbg)
        if OptunaHyperTuning:
            study = optuna.create_study(direction='minimize', study_name='Diamonds XGBoost')
            study.optimize(self.objective, n_trials=100)
            xgb_opt = xgboost.XGBRegressor(**study.best_params, enable_categorical=True, random_state=42)
            xgb_opt.fit(self.x_train_xbg, self.y_train_xbg)
            xgb_opt_pred = xgb_opt.predict(x_test_xbg)
        else:
            xgb = xgboost.XGBRegressor(enable_categorical=True, random_state=42)
            xgb.fit(self.x_train_xbg, self.y_train_xbg)
            xgb_pred = xgb.predict(x_test_xbg)
            MAE=round(mean_absolute_error(y_test_xbg, xgb_pred), 2)
            R2_score=round(r2_score(y_test_xbg, xgb_pred), 4)
            print(MAE)
            print(R2_score)
    
    



        
if __name__ == '__main__':
    Pipeline=AutomatedPipeline2("/home/elios/Desktop/xteam_git/xtream-ai-assignment-developer/data/diamonds.csv")
    #Pipeline.LinearRegression()
    Pipeline.GradientBoosting(OptunaHyperTuning=True)

