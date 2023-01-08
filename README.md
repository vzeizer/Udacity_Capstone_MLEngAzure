# Capstone Project Udacity MLEng w/ Microsoft Azure: Rain Tomorrow in Australia prediction using AzureML

This project consists of using AzureML to find the best model through either AutoML or a customized ML, comparing them, deploying the best model, and consuming it. 
The investigated dataset contains about 10 years of daily weather observations from many locations across Australia.

**RainTomorrow** is the target variable to predict. 
It means -- did it rain the next day, Yes or No? 
This column is Yes if the rain for that day was 1mm or more.

## Project Set Up and Installation
To turn this project into a professional portfolio project, 
you are encouraged to explain how to set up this project in AzureML.

## Dataset

### Overview
I got the data from Kaggle, from [here](https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package?resource=download).
The original columns of the dataset are the following:

-'Date': Date of acquisition;
-'Location': Location where data was collected;
-'MinTemp': Mininum temperature of a given day;
-'MaxTemp': Maximum temperature of a given day;
-'Rainfall': rainfall level;
-'Evaporation': evaporation level in a given day;
-'Sunshine': subshine time;
-'WindGustDir': Wind direction in a given day;
-'WindGustSpeed': Wind speed for a given day;
-'WindDir9am': Wind direction at 9 am;
-'WindDir3pm': Wind direction at 3 pm;
-'WindSpeed9am': Wind Speed at 9 am;
-'WindSpeed3pm': Wind Speed at 3 pm;
-'Humidity9am': Humidity at 9 am;
-'Humidity3pm': Humidity at 3 pm;
-'Pressure9am': Pressure at 9 am;
-'Pressure3pm': Pressura at 3 pm;
-'Cloud9am': Cloud level at 9 am;
-'Cloud3pm': Cloud level at 3 pm;
-'Temp9am': Temperature at 9 am;
-'Temp3pm': Temperature at 3 pm;
-'RainToday': whether it rains today;
-'RainTomorrow': the target variable, whether it will rain tomorrow.

### Data Cleaning

The following procedures were performed for data cleaning of the dataset:

1. drop non-numerical values form "RainTomorrow" and "RainToday";
2. binarize "RainTomorrow" and "RainToday";
3. create date related features;
4. Use SimpleImputer with "mean" strategy for the categorical variables;
5. Use Select KBest with f_classif metric in order to get the 12 most important features;
6. Finally, it returns the engineered features and target variables.

Keep in mind that the SelectKBest method selects the features according to the **k** highest scores, which were chosen as to be 12, and the scoring metric used was the **f_classif** function, which computes ANOVA F-value for the provided sample.


### Task
The goal of this project is to predict next-day rain by training classification models on the target variable *RainTomorrow*.
The features used in this project were shown above.

### Access
*TODO*: Explain how you are accessing the data in your workspace.

## Automated ML
*TODO*: Give an overview of the `automl` settings and configuration you used for this experiment

The AutoML experiment settings are the following:

**automl_settings = {experiment_timeout_minutes=20,
    task='classification',
    primary_metric='accuracy',
    compute_target=cpu_cluster,
    training_data=ds_train,
    label_column_name='y',
    n_cross_validations=5,
    enable_early_stopping=True},
**

briefly explaining it, the AutoML experiment will be run for 20 minutes, the task is classification, the  primary metric to choose the best model is accuracy.
It is also going to use a pre-configured cpu_cluster for the training data which was previously cleaned and prepared for Machine Learning.
The renamed target is 'y', which in this case is "RainTomorrow".
In order to perform hyperparameter tuning, the number of cross validations were chosen as to be 5, and early stopping was enabled in order to run quicker a given model, if the metric (accuracy) keeps not improving.

### Results
*TODO*: What are the results you got with your automated ML model? 
What were the parameters of the model? How could you have improved it?

*TODO* Remember to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Hyperparameter Tuning
I chose the Logistic Regression to test as model in this dataset because it is a rather simple model, and it has some kind of success in modeling classification problems.
Moreover, it is a "light" model, which would be a fast model to deploy and to consume.
The parameters used for Hyperparameter Tuning of the Logistic Regression model was chosen as to be "C" and "max_iter", the latter being the inverse of the regularization strength and the former the maximum number of iterations for the algorithm to converge. 
The values  of "C" was chosen as to be in [0.01,0.1,1.0,10.,100.] because ranges huge regularization strength to very low.
The "max_iter" parameter as chosen to be in [100,500,1000] because it ranges from few iterations up to a high number.

The Bayesian parameter optimization was chosen because it provides, in general, results as good as a grid search sampling, but in a smarter and faster way.

### Results
*TODO*: What are the results you got with your model? 
What were the parameters of the model? 
How could you have improved it?



*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.



## Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

## Screen Recording
*TODO* Provide a link to a screen recording of the project in action. 
Remember that the screencast should demonstrate:
- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response

## Standout Suggestions

Some improvements can be performed in the project, in order to enhance it:

1. Run the AutoML experiment for longer times, for instance, 60 minutes;
2. Use other Machine Learning classification models for the customized training model, for instance, KNN and Decision Tree classifiers;
3. perform more feature engineering and try other data cleaning approaches in the customized training model, in order to improve results;
4. Select a larger number (more than 12) of **k** best features obtained from the SelectKBest in the data cleaning process;
5. remove outliers and perform PCA analysis in order to reduce dimensionality of the dataset instead of SelectKBest;
6. Deploy ONNX compatible models obtained from the AutoML or the customized model.





