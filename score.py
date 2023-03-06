# the following code was take from the following URL:
# "https://learn.microsoft.com/en-us/azure/machine-learning/how-to-deploy-online-endpoints?tabs=python#define-an-entry-script"

import os
import logging
import json
import numpy
from azureml.core import Model
import joblib
import pandas as pd

#os.environ['AZUREML_MODEL_DIR'] = "/mnt/batch/tasks/shared/LS_root/mounts/clusters/vzeizer1/code/Users/vzeizer"

def init():
	"""
	This function is called when the container is initialized/started, 
	typically after create/update of the deployment.
	You can write the logic here to perform init operations like 
	caching the model in memory
	"""
	global model
	# AZUREML_MODEL_DIR is an environment variable created during deployment.
	# It is the path to the model folder (./azureml-models/$MODEL_NAME/$VERSION)
	# Please provide your model's folder name if there is one
	# should care about the path!
#	model_path = Model.get_model_path('bestmodel_all.pkl')
#	model_path = os.path.join(model_path, "bestmodel_all.pkl")
	model_path=os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'bestmodel_all.joblib')

	# deserialize the model file back into a sklearn model
	model = joblib.load(model_path)
	logging.info("Init complete")


def run(data):
	"""
	This function is called for every invocation of the endpoint to perform the actual scoring/prediction.
	In the example we extract the data from the json input and call the scikit-learn model's predict()
	method and return the result back
	"""
	try:
		logging.info("model 1: request received")
		data_1 = json.loads(data)#["data"]
		df=pd.DataFrame.from_dict(data_1)
		print('aquiiiiiii')
		result = model.predict(df)
		logging.info("Request processed")
		return result.tolist()
	except Exception as e:
		print('error:{0}'.format(e))
