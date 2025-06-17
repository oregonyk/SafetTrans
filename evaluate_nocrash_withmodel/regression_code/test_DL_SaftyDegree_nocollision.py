import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from evaluate_nocrash_withmodel.regression_code.RegressionModel import RegressionModelTET, RegressionModelSaftyDegree
from evaluate_nocrash_withmodel.utils.data_processing import preprocess_data, readcsv
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
def load_model(model_path, input_dim):
    model = RegressionModelSaftyDegree(input_dim=input_dim)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def predict(model, data_path, label_name):
    data = pd.read_csv(data_path)
    label_pos = {"SaftyDegree": 13, "TIT": 15, "TET": 16, "Dece": 17}
    label_index = label_pos[label_name]
    exclude_value = 0

    filtered_data = data[data[label_name] > exclude_value]
    X = filtered_data.iloc[:, :12].values
    y = filtered_data.iloc[:, label_index].values

    scaler = MinMaxScaler()
    scaler_path = "retrainedRegressionNocollisionSaftyDegreeMinMaxScaler_1.pt"
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
def getNocollision(x,modelPath,MinMaxScalerPath):
    model = load_model(modelPath,input_dim=12)
    scaler = MinMaxScaler()
    scaler = joblib.load(MinMaxScalerPath)
    X_scaled = scaler.transform(x)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    with torch.no_grad():
        predictions = model(X_tensor).detach().numpy()
    #print(predictions[0][0])
    return predictions[0][0]
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="../dataset/dataset_safevar/CARLA_SUN.csv",
                        help="Path to the data file")
    parser.add_argument("--label_name", type=str, default="SaftyDegree", help="Label name to be expected")
    parser.add_argument("--model_path", type=str, default="../model/Torch/NocollisionCARLA_SUN_SaftyDegree_BEST.pth",
                        help="Path to the trained model file")
    args = parser.parse_args()
    if not os.path.exists(args.data_path):
        raise RuntimeError("Incorrect data_path")

    model = load_model(args.model_path, input_dim=12)
    predict(model, args.data_path, args.label_name)
