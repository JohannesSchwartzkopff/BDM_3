#import os
#import warnings
import sys

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.sklearn
#import logging
from urllib.parse import urlparse

import pandas as pd
import numpy as np
import pickle

#os.environ["MLFLOW_TRACKING_URI"] = f"http://training.itu.dk:5000/"

# ## Azure stuff
# from azure.ai.ml import MLClient
# from azure.identity import DefaultAzureCredential
# from azureml.core import Workspace
# # Enter details of your subscription
# subscription_id = "a5152d52-6a98-41f9-92ea-0a99dcc6347b"
# resource_group = "Group1"
# workspace_name = "Assignment3"
# ml_client = MLClient(credential=DefaultAzureCredential(),
#                      subscription_id=subscription_id, 
#                      resource_group_name=resource_group)
# azureml_mlflow_uri = ml_client.workspaces.get(workspace_name).mlflow_tracking_uri
# mlflow.set_tracking_uri(azureml_mlflow_uri)
# # Set name
# mlflow.set_experiment("Orkney_wind_experiment")

#Set tracking URI
#mlflow.set_tracking_uri("azureml://northeurope.api.azureml.ms/mlflow/v1.0/subscriptions/a5152d52-6a98-41f9-92ea-0a99dcc6347b/resourceGroups/Group1/providers/Microsoft.MachineLearningServices/workspaces/Assignment3")
#experiment_ID = mlflow.set_experiment("Orkney_wind_experiment")

# This code was supplied by the teaching team of Big Data Management.
# logging.basicConfig(level=logging.WARN)
# logger = logging.getLogger(__name__)

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

if __name__ =="__main__":
   # warnings.filterwarnings("ignore")

    # For using own data

    data_file_name = '/Users/johannesschwartzkopff/Big_Data_Mngmnt/Assignment_3/orkney_wind/official_data.csv' #Default
    if(len(sys.argv)>2): #If given, set to given.
        data_file_name = sys.argv[2]
    main_df = pd.read_csv(data_file_name)

    X = main_df[['Direction','Speed']]
    Y = main_df[['Total']]

    ##Taking samples of the data for training of model and testing of the model
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42, shuffle=False)

    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.preprocessing import PolynomialFeatures

    enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
    ct = ColumnTransformer([("encoder transformer", enc, ["Direction"])], remainder="passthrough")
    scaler = StandardScaler()
    poly_degree = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    poly = PolynomialFeatures(degree=poly_degree, include_bias=False)
    poly_reg = LinearRegression()


    with mlflow.start_run():
        #for debugging
        client = mlflow.tracking.MlflowClient()
        data_track = client.get_run(mlflow.active_run().info.run_id).data
        print(client)
        print(data_track)

        # Define the pipeline steps. 
        pipeline_poly_scaled = Pipeline(steps=[
                                    ("ct", ct),    # step 1 in pipeline: Transform the columns we specified above (onehote encode state)
                                    ("scale",scaler),
                                    ("preprocessor",poly),
                                    ("lr", poly_reg)    # step 2 in pipeline: Fit the logistic model using the transformed data
                                ])

        pipeline_poly_scaled = pipeline_poly_scaled.fit(x_train, y_train)

        pred_y = pipeline_poly_scaled.predict(x_test)

        (rmse, mae, r2) = eval_metrics(y_test, pred_y)

        print("Polynomial regression model (poly_degree=%f):" % (poly_degree))
        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)
        print("  R2: %s" % r2)

        
        #mlflow.log_param("poly_degree",poly_degree)
        #mlflow.log_metric("rmse",rmse)
        #mlflow.log_metric("r2",r2)
        #mlflow.log_metric("mae",mae)


        

        #tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        # Model registry does not work with file store
        #if tracking_url_type_store != "file":

            # Register the model
            # There are other ways to use the Model Registry, which depends on the use case,
            # please refer to the doc for more information:
            # https://mlflow.org/docs/latest/model-registry.html#api-workflow
        #    mlflow.sklearn.log_model(pipeline_poly_scaled, "PolynomialRegressionModel_degree3")

