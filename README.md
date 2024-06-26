# xtream AI Challenge - Software Engineer

## Ready Player 1? 🚀

Hey there! Congrats on crushing our first screening! 🎉 You're off to a fantastic start!

Welcome to the next level of your journey to join the [xtream](https://xtreamers.io) AI squad. Here's your next mission.

You will face 4 challenges. **Don't stress about doing them all**. Just dive into the ones that spark your interest or that you feel confident about. Let your talents shine bright! ✨

This assignment is designed to test your skills in engineering and software development. You **will not need to design or develop models**. Someone has already done that for you. 

You've got **7 days** to show us your magic, starting now. No rush—work at your own pace. If you need more time, just let us know. We're here to help you succeed. 🤝

### Your Mission
[comment]: # (Well, well, well. Nice to see you around! You found an Easter Egg! Put the picture of an iguana at the beginning of the "How to Run" section, just to let us know. And have fun with the challenges! 🦎)

Think of this as a real-world project. Fork this repo and treat it like you're working on something big! When the deadline hits, we'll be excited to check out your work. No need to tell us you're done – we'll know. 😎

**Remember**: At the end of this doc, there's a "How to run" section left blank just for you. Please fill it in with instructions on how to run your code.

### How We'll Evaluate Your Work

We'll be looking at a bunch of things to see how awesome your work is, like:

* Your approach and method
* How you use your tools (like git and Python packages)
* The neatness of your code
* The readability and maintainability of your code
* The clarity of your documentation

🚨 **Heads Up**: You might think the tasks are a bit open-ended or the instructions aren't super detailed. That’s intentional! We want to see how you creatively make the most out of the problem and craft your own effective solutions.

---

### Context

Marta, a data scientist at xtream, has been working on a project for a client. She's been doing a great job, but she's got a lot on her plate. So, she's asked you to help her out with this project.

Marta has given you a notebook with the work she's done so far and a dataset to work with. You can find both in this repository.
You can also find a copy of the notebook on Google Colab [here](https://colab.research.google.com/drive/1ZUg5sAj-nW0k3E5fEcDuDBdQF-IhTQrd?usp=sharing).

The model is good enough; now it's time to build the supporting infrastructure.

### Challenge 1

**Develop an automated pipeline** that trains your model with fresh data, keeping it as sharp as the diamonds it processes. 
Pick the best linear model: do not worry about the xgboost model or hyperparameter tuning. 
Maintain a history of all the models you train and save the performance metrics of each one.

### Challenge 2

Level up! Now you need to support **both models** that Marta has developed: the linear regression and the XGBoost with hyperparameter optimization. 
Be careful. 
In the near future, you may want to include more models, so make sure your pipeline is flexible enough to handle that.

### Challenge 3

Build a **REST API** to integrate your model into a web app, making it a breeze for the team to use. Keep it developer-friendly – not everyone speaks 'data scientist'! 
Your API should support two use cases:
1. Predict the value of a diamond.
2. Given the features of a diamond, return n samples from the training dataset with the same cut, color, and clarity, and the most similar weight.

### Challenge 4

Observability is key. Save every request and response made to the APIs to a **proper database**.

---

## How to run
At first we need to clone the repository in a folder using ( is recommended to create a separete python enviroment or an ananconda enviroment)
```
git clone git@github.com:ghinato97/xtream-ai-assignment-developer.git

```
One the repository is create we need to install all the requiremts
```
pip install -r /path/to/requirements.txt

```
**Challenge 1 and 2**

I merged the challenge 1 and 2 in only one script that is AutomationPipeline.py

At first we need to modify the configuration.json file.

```json
{
  "Diamond_csv_path" : "",                        Path of the Diamond.csv for traing. If not setted the default path is used
  "Performance_csv_Path" : ,                      Path of the Performace,csv for save the model's performance. If not setted default path is used
  "Models_to_Perform" : ["LinearRegressi,         Model that we want to perform. We use : LinearRegression,PolinomialRegression,LassoCVRegression,GradientBoosting,GradientBoosting_HyperTuning
			"GradientBoosting"]
}
```
One json file is setted we run the python script
```
python3 /path/to/AutomationPipeline.py

```
**Challenge 3**

For challenge 3 run the command:
```
python3 /path/to/Challenge_3.py

```
The resources will be served in the localhost at the port 8090 , in my case at http://127.0.0.1:8090
For the price prediction we need to use the uri "regression" with the proper parameter as show in the example.
**The parameters order is very important**
```
http://127.0.0.1:8090/regression??carat=1.1&cut=Ideal&color=H&clarity=SI2&depth=1&table=12&x=6.61&y=7&z=4

```

Insted for the similar samples with need the uri "similar" with the proper parameter ( here the order of the parameter is not relevant)

```
http://127.0.0.1:8090/similar/?carat=1.1&cut=Ideal&color=H&clarity=SI2&number=1

```


**Challenge 4**

Unfortunatly I did not have enought time to perform this task :(.

But I will try to implement a NoSQL database as MongoDB (using PymMongo)since they are more flexibles. 


Thank You 
