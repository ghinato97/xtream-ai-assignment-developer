#Elios Ghinato 19/06/2024  
#Challenge 3



#Cherrypy as Web Framework
import cherrypy 
import pandas as pd
import pickle
from API_CommonFunction import *



class RestApi(object):
    exposed=True
    def __init__(self,DataSetPath,voidDataSet,ModelPath):
        #parameter that need to be used for the search of similarity
        similarity_label=["carat","cut", "color","clarity","number"]
        similarity_label.sort()
        self.similarityLabel=similarity_label

        #parameter that nned to be used for the regression
        regression_labels=["carat","cut","color","clarity","depth","table","x","y","z"]
        regression_labels.sort()
        self.regressionLabel=regression_labels

        #Load Dataset
        self.DataSet=pd.read_csv(DataSetPath)
        self.voidDataset=pd.read_csv(voidDataSet)


        #Load Predictive Model
        with open('/home/elios/Desktop/xteam_git/xtream-ai-assignment-developer/Models/GradientBoosting.pkl', 'rb') as f:
            self.RegressionModel = pickle.load(f)




    def GET(self,*uri,**params):
        if uri[0] == 'regression':
            key=list(params.keys())
            key.sort()
            if (key == self.regressionLabel):
                price = self.Regression(params)
                return str(price)
            else:
                return "parameter input error"

        if uri[0] == "similar":
            key=list(params.keys())
            key.sort()
            if (key == self.similarityLabel):
                htmlDataFrame = self.SimilaritySearch(params)
                return htmlDataFrame
            else:
                return "Parameter Input Error"
        else:
            return "uri not correct"
    
    # the regression is done with the best model of 
    
    
    def Regression(self,params):
       errorFlag , errorParameter = CheckParamaterRegression(self.DataSet,params)
       if errorFlag:
            strinError = "Error with " + errorParameter + " parameters input"
            return strinError
       DataSet=CreateDataSet(params)
       regressionModel = LoadModel()
       price=regressionModel.predict(DataSet)
       print(price)
       return price[0]
       



    def SimilaritySearch(self,params):

        errorFlag , errorParameter = CheckParamaterSimilarity(self.DataSet,params)

        if errorFlag:
            strinError = "Error with " + errorParameter + " parameters input"
            return strinError
        
        itemNumber2search = int(params["number"])
        carat = float(params["carat"])
        cut = params["cut"]
        color = params["color"]
        clarity = params["clarity"]

        miniDataSet = self.DataSet[(self.DataSet.cut==cut)&(self.DataSet.color==color)&(self.DataSet.clarity==clarity)]
        miniDataSet = miniDataSet.iloc[(miniDataSet['carat']-carat).abs().argsort()[:itemNumber2search]]

        if miniDataSet.empty:
            strinError="No match fund"
            return strinError
        else:
            return miniDataSet.to_html()


  


if __name__=="__main__":

  dataSetPath="/home/elios/Desktop/xteam_git/xtream-ai-assignment-developer/data/diamonds.csv"
  voidDataSet="/home/elios/Desktop/xteam_git/xtream-ai-assignment-developer/data/DataFrameVoid.csv"
  modelPath=""
  conf={
          '/':{
                'request.dispatch':cherrypy.dispatch.MethodDispatcher(),
                'tool.session.on':True
               }
        }
  cherrypy.config.update({'server.socket_port':8090})
  cherrypy.quickstart(RestApi(dataSetPath,voidDataSet,modelPath),'/',conf)