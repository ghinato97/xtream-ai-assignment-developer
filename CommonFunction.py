#Common Function
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import time
import pickle



def LoadDataset_(DataSetOriginal,CollinearityFilter=False,CategoricalFileter=False):
    DataSet=DataSetOriginal
    if CollinearityFilter: #For LinearRegression
        DataSetCollinearity = DataSet.drop(columns=['depth', 'table', 'y', 'z'])
        return  DataSetCollinearity
    if CategoricalFileter: #For xgboost
        DataSet_xgb=DataSet
        DataSet_xgb['cut'] = pd.Categorical(DataSet['cut'], categories=['Fair', 'Good', 'Very Good', 'Ideal', 'Premium'], ordered=True)
        DataSet_xgb['color'] = pd.Categorical(DataSet['color'], categories=['D', 'E', 'F', 'G', 'H', 'I', 'J'], ordered=True)
        DataSet_xgb['clarity'] = pd.Categorical(DataSet['clarity'], categories=['IF', 'VVS1', 'VVS2', 'VS1', 'VS2', 'SI1', 'SI2', 'I1'], ordered=True)
        return DataSet_xgb




def TrainTestSplit_(Dataset_,Dummy=False,StandarScaler=False):
        if Dummy:
            DataSet = pd.get_dummies(Dataset_, columns=['cut', 'color', 'clarity'], drop_first=True)
        
        DataSet=Dataset_
        x = DataSet.drop(columns='price')
        y = DataSet.price
        if StandarScaler:
            scaler = StandardScaler()
            x_scaled = scaler.fit_transform(x)
            return (train_test_split(x_scaled, y, test_size=0.2, random_state=42))
        else:            
            return (train_test_split(x, y, test_size=0.2, random_state=42))


#Create or modify  a csv file  with models parameters performance
def PerformanceFile_(SavePath,DictParameter):
    if os.path.isfile(SavePath):
        ModelPerformanceFile = pd.read_csv(SavePath)
        ModelPerformanceFile = ModelPerformanceFile._append(DictParameter,ignore_index=True)
        ModelPerformanceFile.to_csv(SavePath,index=False)
    else:
        ModelPerformanceFile = pd.DataFrame(DictParameter,index=[0])
        ModelPerformanceFile.to_csv(SavePath,index=False)


def SaveModel(model,name):
    generalDirectory = os.path.dirname(os.path.abspath(__file__))    
    ModelsFolderPath=os.path.join(generalDirectory,"Models")
    nameModel=name+".pkl"
    ModelsPath=os.path.join(ModelsFolderPath,nameModel)
    print(ModelsPath)
    
    #if not present create the "Models" folder
    os.makedirs(ModelsFolderPath, exist_ok=True)


    with open(ModelsPath,'wb') as f:
        pickle.dump(model,f)