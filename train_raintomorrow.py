from sklearn.linear_model import LogisticRegression
import argparse
import os
import numpy as np
from sklearn.metrics import mean_squared_error
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from azureml.core.run import Run
from azureml.data.dataset_factory import TabularDatasetFactory
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif


def data_transform(df):
	"""
	This function process the data, cleaning it, and outputing the training and label data.
	Input: df: raw data
	Output: df[numeric_cols+encoded_cols]: training data features numeric and encoded.
			y_df: the binarized target variable
	"""
    # reads the data
#    df = pd.read_csv("./weatherAUS.csv")
    # drops na in columns ['RainToday','RainTomorrow']
    df=df.to_pandas_dataframe().dropna(subset=['RainToday','RainTomorrow'])
    
    y = df.pop("RainTomorrow").apply(lambda x: 1 if x == "Yes" else 0)
    df['RainToday']=df['RainToday'].apply(lambda x: 1 if x == "Yes" else 0)
    
    # creates date time related features
    year=pd.to_datetime(df.Date).dt.year
    month=pd.to_datetime(df.Date).dt.month
    day=pd.to_datetime(df.Date).dt.day
    
    df['year']=year
    df['month']=month
    df['day']=day
    
    #train_df=df[df['year']<=2015]
    #test_df=df[df['year']>2015]
    # drops the Date Feature
    df=df.drop(['Date'],axis=1).copy()
    # separating the columns in numeric and categorical
    numeric_cols=df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols=df.select_dtypes(include=object).columns.tolist()
    # using an Imputer to fill with the mean
    Imputer= SimpleImputer(strategy='mean')
    Imputer.fit(df[numeric_cols])
    # imputing the numeric columns
    df[numeric_cols]=Imputer.transform(df[numeric_cols])    
    # replace the nulls in the categorical variable with the mode
    df[categorical_cols] = df[categorical_cols].fillna(df[categorical_cols].mode().iloc[0])
    # using OneHotEncoder to deal with categorical data
    encoder = OneHotEncoder(sparse=False,handle_unknown='ignore')
    encoder.fit(df[categorical_cols])
    # list of encoded columns
    encoded_cols = list(encoder.get_feature_names(categorical_cols))
    # transforming encoded columns
    df[encoded_cols]=encoder.transform(df[categorical_cols])
    
    X=df[numeric_cols+encoded_cols].copy()
    
    selector = SelectKBest(f_classif, k=12)
    selector.fit_transform(X, y)
    
    cols = selector.get_support(indices=True)
    X_new = X.iloc[:,cols]
    
    return X_new,y

def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('--C', type=float, default=1.0, help="Inverse of regularization strength. Smaller values cause stronger regularization")
    parser.add_argument('--max_iter', type=int, default=100, help="Maximum number of iterations to converge")

    args = parser.parse_args()

    run = Run.get_context()

    run.log("Regularization Strength:", np.float(args.C))
    run.log("Max iterations:", np.int(args.max_iter))

    # TODO: Create TabularDataset using TabularDatasetFactory
    # Data is located at:
    # "https://raw.githubusercontent.com/vzeizer/Udacity_Capstone_MLEngAzure/master/weatherAUS.csv"
    path='https://raw.githubusercontent.com/vzeizer/Udacity_Capstone_MLEngAzure/master/weatherAUS.csv'
    ds = TabularDatasetFactory.from_delimited_files(path)### YOUR CODE HERE ###
    
    x, y = clean_data(ds)

    # TODO: Split data into train and test sets.

    ### YOUR CODE HERE ###
    train_size=0.7
    x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=train_size,
    random_state=42)
    
    model = LogisticRegression(C=args.C, max_iter=args.max_iter).fit(x_train, y_train)

    accuracy = model.score(x_test, y_test)
    run.log("Accuracy", np.float(accuracy))

if __name__ == '__main__':
    main()


