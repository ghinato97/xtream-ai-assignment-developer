import pandas as pd
import os
import pickle
from Challenge_2 import ModelRegression

    
def CreateDataSet(params):
    # carat', 'cut', 'color', 'clarity', 'depth', 'table', 'x', 'y', 'z' is the order of the columns
    params["carat"] = float(params["carat"])
    params["cut"] = str(params["cut"])
    params["color"] = str(params["color"])
    params["clarity"] = str(params["clarity"])
    params["depth"] = float(params["depth"])
    params["table"] =float(params["table"])
    params["x"] = float(params["x"])
    params["y"] = float(params["x"])
    params["z"] = float(params["z"])
    
    DataSet=pd.DataFrame(params,index=[0])
    print(DataSet)

    DataSet['cut'] = pd.Categorical(DataSet['cut'], categories=['Fair', 'Good', 'Very Good', 'Ideal', 'Premium'], ordered=True)
    DataSet['color'] = pd.Categorical(DataSet['color'], categories=['D', 'E', 'F', 'G', 'H', 'I', 'J'], ordered=True)
    DataSet['clarity'] = pd.Categorical(DataSet['clarity'], categories=['IF', 'VVS1', 'VVS2', 'VS1', 'VS2', 'SI1', 'SI2', 'I1'], ordered=True)
    print(DataSet)
    return DataSet




def CheckParamaterSimilarity(DataSet,Paramater):
    cutValues = list(DataSet['cut'].unique())
    cutValues = [x.upper() for x in cutValues ]
    if not (Paramater["cut"].upper() in cutValues):
        return (True,"cut")

    clarityValues = list(DataSet['clarity'].unique())
    clarityValues = [x.upper() for x in clarityValues]
    if not (Paramater["clarity"].upper() in clarityValues):
        return (True,"clarity")

    colorValues = list(DataSet['color'].unique())
    colorValues = [x.upper() for x in colorValues]
    if not (Paramater["color"].upper() in colorValues):
        return (True,"color")
    
    try:
        itemNumber2search = int(Paramater["number"])
    except:
        return (True,"number")
    
    try:
        carat = float(Paramater["carat"])
    except:
        return (True,"carat")
    
    return (False, "")


def CheckParamaterRegression(DataSet,Paramater):
    cutValues = list(DataSet['cut'].unique())
    cutValues = [x.upper() for x in cutValues ]
    if not (Paramater["cut"].upper() in cutValues):
        return (True,"cut")

    clarityValues = list(DataSet['clarity'].unique())
    clarityValues = [x.upper() for x in clarityValues]
    if not (Paramater["clarity"].upper() in clarityValues):
        return (True,"clarity")

    colorValues = list(DataSet['color'].unique())
    colorValues = [x.upper() for x in colorValues]
    if not (Paramater["color"].upper() in colorValues):
        return (True,"color")
    
    try:
        carat = float(Paramater["carat"])
    except:
        return (True,"carat")
    
    try:
        carat = float(Paramater["x"])
    except:
        return (True,"x")
    
    try:
        carat = float(Paramater["y"])
    except:
        return (True,"y")
    
    try:
        carat = float(Paramater["z"])
    except:
        return (True,"z")
    
    try:
        carat = float(Paramater["table"])
    except:
        return (True,"table")
    
    try:
        carat = float(Paramater["depth"])
    except:
        return (True,"depth")
    
    return (False, "")


# If the model is not find in the path, it try in the default model folder and if there is no model 
# saved train one and save in the default folder 
def LoadModel(ModelPath=""):

    generalDirectory = os.path.dirname(os.path.abspath(__file__))
    ModelDefaultPath=os.path.join(generalDirectory,"Models/GradientBoosting.pkl")
    print(ModelDefaultPath)

    if os.path.exists(ModelPath):
        with open(ModelPath, 'rb') as file:
            model = pickle.load(file)
            return model
        

    if os.path.exists(ModelDefaultPath):
        with open(ModelDefaultPath, 'rb') as file:
            model = pickle.load(file)
            return model
        
    else:
        ModelRegression().GradientBoosting()
        LoadModel()
    
    
    


