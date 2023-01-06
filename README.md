*NOTE:* This file is a template that you can use to create the README for your project. The *TODO* comments below will highlight the information you should be sure to include.

# Your Project Title Here

*TODO:* Write a short introduction to your project.
Content

This project consists of using AzureML to find the best model through either AutoML or a customized ML, comparing them, and deploying the best model and consume it. 

The investigated dataset contains about 10 years of daily weather observations from many locations across Australia.

*RainTomorrow* is the target variable to predict. 
It means -- did it rain the next day, Yes or No? 
This column is Yes if the rain for that day was 1mm or more.

## Project Set Up and Installation
*OPTIONAL:* If your project has any special installation steps, this is where you should put it. 
To turn this project into a professional portfolio project, 
you are encouraged to explain how to set up this project in AzureML.

## Dataset

### Overview
*TODO*: Explain about the data you are using and where you got it from.

### Task
*TODO*: Explain the task you are going to be solving with this dataset and the features you will be using for it.
The goal of this project is to predict next-day rain by training classification models on the target variable *RainTomorrow*.


### Access
*TODO*: Explain how you are accessing the data in your workspace.

## Automated ML
*TODO*: Give an overview of the `automl` settings and configuration you used for this experiment

### Results
*TODO*: What are the results you got with your automated ML model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Hyperparameter Tuning
*TODO*: What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search


### Results
*TODO*: What are the results you got with your model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

## Screen Recording
*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate:
- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response

## Standout Suggestions

Some improvements can be performed in the project, in order to enhance it:

1. Run the AutoML experiment for longer times, for instance, 60 minutes;
2. Use other Machine Learning classification models for the customized training model, for instance, KNN and Decision Tree classifiers;
3. perform more feature engineering and try other data cleaning approaches in the customized training model, in order to improve results;







