import os
import joblib
import numpy as np
import pandas as pd
import argparse
from dkube.sdk import *

model_dir = "/model"
data_dir = "/data"

def predict():
    dkubefs = DkubeFeatureSet()
    test_df = dkubefs.read_features(path = data_dir)
    testdf_tmp = test_df
    df = testdf_tmp.drop("PassengerId", 1)
    #df = testdf_tmp.drop(["PassengerId","Survived"], 1)
    df = pd.DataFrame(df).fillna(df.mean())
    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    predictions = model.predict(df)
    output = pd.DataFrame({'PassengerId': test_df.PassengerId, 'Survived': predictions})
    output.to_csv("/tmp/prediction.csv", index=False)
    print("predictions generated.")

if __name__ == "__main__":
    predict()
