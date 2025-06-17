import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from evaluate_nocrash_withmodel.regression_code.RegressionModel import BinaryModelSaftyDegree
from evaluate_nocrash_withmodel.utils.data_processing import preprocess_data, readcsv
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
def load_model(model_path, input_dim):
    model = BinaryModelSaftyDegree(input_dim=input_dim)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def predict(model, data_path, MinMaxScalerPath,label_name):
    data = pd.read_csv(data_path)
    label_pos = {"SaftyDegree": 13, "TIT": 15, "TET": 16, "Dece": 17}
    label_index = label_pos[label_name]
    exclude_value = 0

    filtered_data = data[data[label_name] != exclude_value]
    X = filtered_data.iloc[:, :12].values
    y = filtered_data.iloc[:, label_index].values

    scaler = MinMaxScaler()
    scaler_path = MinMaxScalerPath
    scaler = joblib.load(scaler_path)
    X_scaled = scaler.transform(X)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    with torch.no_grad():
        predictions = model(X_tensor).detach().numpy()
    # Inverse transform predictions to original scale
    #reshaped_predictions = predictions.reshape(-1, 1)
    original_predictions = predictions
    for i in range(0,1000,10):
        print(original_predictions[i],y[i])
    #print("Predictions:", original_predictions)
def getSaftyDegree(x,modelPath,MinMaxScalerPath):
    model = BinaryModelSaftyDegree(input_dim=12)
    model.load_state_dict(torch.load(modelPath))
    model.eval()
    scaler = joblib.load(MinMaxScalerPath)
    X_scaled = scaler.transform(x)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    with torch.no_grad():
        predictions = model(X_tensor).detach().numpy()
    #return 0 if predictions[0][0]<0.5 else 1\
    return predictions[0][0]
if __name__ == "__main__":

    data_path = "../newDataset/Scenario2.csv"
    label_name = "SaftyDegree"
    modelPath = "../retraining/retrainedRegressionSaftyDegreeTorch_1.pt"
    MinMaxScalerPath = "../regression_code/retrainedRegressionSaftyDegreeMinMaxScaler_1.pt"

    #model = load_model(modelPath, input_dim=12)
    #predict(model, data_path,MinMaxScalerPath, label_name)
    modelPath = "../retraining/retrainedRegressionSaftyDegreeTorch_1.pt"
    modelPath = '../model/Torch/CARLA_SUN_SaftyDegree_BEST.pth'
    MinMaxScalerPath = "retrainedRegressionSaftyDegreeMinMaxScaler_1.pt"
    x = [[5733.597, 0.125, 1.851, 0.21, 0.347, 10., 2627.511, 0.232, 3.318, 0.296, 35.5, 1634.117]]
    ans = getSaftyDegree(x, modelPath, MinMaxScalerPath)
    print(ans)