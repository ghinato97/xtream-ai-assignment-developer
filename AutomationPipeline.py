#Automatin Pipeline

from Challenge_2 import ModelRegression
import json
import os


class AutomationPipeline(ModelRegression):
    def __init__(self,ConfigurationPath =""):

        if ConfigurationPath == "":
            generalDirectory = os.path.dirname(os.path.abspath(__file__))
            jsonPath = os.path.join(generalDirectory,"configuration.json")
            f = open(jsonPath)
            jsonFile = json.loads(f.read())
            f.close()
        else:
            f = open(ConfigurationPath)
            jsonFile = json.loads(f.read())
            f.close()

        DiamondData = jsonFile["Diamond_csv_path"]
        PerformancePath = jsonFile["Performance_csv_Path"]
        self.model2Train = jsonFile["Models_to_Perform"]

        super().__init__(DiamondData, PerformancePath)

        self.Start()


    def Start(self):
        for model in self.model2Train:
            print(model)
            if model == "LinearRegression":
                self.LinearRegression()
                continue
            if model == "PolinomialRegression":
                self.PolinomialRegression()
                continue
            if model == "LassoCVRegression":
                self.LassoCVRegression()
                continue
            if model == "GradientBoosting":
                self.GradientBoosting()
                continue
            if model == "GradientBoosting_HP":
                self.GradientBoosting(OptunaHyperTuning=False)
                continue
            else:
                print("Model not available")
    



if __name__ == '__main__':
    AutomationPipeline()