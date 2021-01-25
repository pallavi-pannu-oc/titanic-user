import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
import requests, os
import argparse
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import mlflow
from dkube.sdk import *

inp_path = "/titanic-train"
out_path = "/model"
test_path = "/titanic-test"


if __name__ == "__main__":

    ########--- Parse for parameters ---########

    parser = argparse.ArgumentParser()
    parser.add_argument("--url", dest="url", default=None, type=str, help="setup URL")
    parser.add_argument("--train_fs", dest="train_fs", required=True, type=str, help="featureset")
    parser.add_argument("--test_fs", dest="test_fs", required=True, type=str, help="featureset")

    global FLAGS
    FLAGS, unparsed = parser.parse_known_args()
    dkubeURL = FLAGS.url
    train_fs = FLAGS.train_fs
    test_fs = FLAGS.test_fs

    ########--- Read features from input FeatureSet ---########

    # Featureset API
    authToken = os.getenv("DKUBE_USER_ACCESS_TOKEN")
    # Get client handle
    api = DkubeApi(URL=dkubeURL, token=authToken)

    # Read features
    feature_df = api.read_featureset(name = train_fs)  # output: data

    train, val = train_test_split(feature_df, test_size=0.2)
    ########--- Train ---########

    # preparing input output pairs
    y = train["Survived"].values
    x = train.drop(["PassengerId","Survived"], 1).values

    # Training random forest classifier
    model_RFC = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
    model_RFC.fit(x, y)
    predictions = model_RFC.predict(x)

    ########--- Log metrics to DKube ---########

    # Calculating accuracy
    accuracy = accuracy_score(y, predictions)
    # logging acuracy to DKube
    mlflow.log_metric("accuracy", accuracy)

    ########--- Write model to DKube ---########

    # Exporting model
    filename = os.path.join(out_path, "model.joblib")
    joblib.dump(model_RFC, filename)

    # Writing val data
    val.to_csv(os.path.join(out_path, "val.csv"), index=False)