import argparse
import os
import math

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
#from ..regression_code.RegressionModel import RegressionModelTIT
from evaluate_nocrash_withmodel.regression_code.RegressionModel import RegressionModelTIT
# Define your model architecture

def readcsv(path,label,MinMaxScalerPath):
    file_path = path
    data = pd.read_csv(file_path)  # 从 data.csv 文件中读取数据
    label_pos = {"SaftyDegree": 12, "TIT": 14, "TET": 13}#这里对应的是新加的TIT位置，与safevar中不同
    filtered_data = data
    X = filtered_data.iloc[:, :12].values
    label_index = label_pos[label]
    y = filtered_data.iloc[:, label_index].values
    scaler = joblib.load(MinMaxScalerPath)
    X_scaled = scaler.fit_transform(X)
    #print(y)
    return X_scaled, y

def reTrainTITModel(modelPath,MinMaxScalerPath,data_path,label_name):
    # Load and preprocess the data
    # MinMaxScalerPath = "../model/minmax/minmax_scaler_model_best.pkl"
    X, y = readcsv(data_path, label_name,MinMaxScalerPath)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Convert data to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
    print(X_train.shape[1])
    # Define your custom model
    model = RegressionModelTIT(input_dim=X_train.shape[1])

    # Load pretrained weights (assuming 'pretrained_model.pth' contains the weights)
    # pretrained_weights_path = "../model/Torch/CARLA_SUN_BEST.pth"
    pretrained_weights_path = modelPath
    pretrained_dict = torch.load(pretrained_weights_path)
    model_dict = model.state_dict()
    # Filter out unnecessary keys and load pretrained weights
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    layers_to_freeze = [0, 1, 2, 3, 4]  # 索引从0开始，对应需要冻结的层
    layer_count = 0

    for name, param in model.named_parameters():
        if layer_count not in layers_to_freeze:
            param.requires_grad = False
        layer_count += 1

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)


    num_epochs = 200
    best_test_rmse = float('inf')  # 初始化为正无穷大
    best_model_path = ""

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs.squeeze(), y_train_tensor)
        loss.backward()
        optimizer.step()

        # Evaluate the model
        model.eval()
        with torch.no_grad():
            train_predictions = model(X_train_tensor).detach().numpy()
            test_predictions = model(X_test_tensor).detach().numpy()
            train_rmse = np.sqrt(mean_squared_error(y_train, train_predictions))
            test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
            if (epoch % 10 == 0):
                print(f"Epoch [{epoch + 1}/{num_epochs}], Train RMSE: {train_rmse:.4f}, Test RMSE: {test_rmse:.4f}")

            # Output some predictions and true values
            num_samples = 2
            #print("Sample predictions and true values:")
            for i in range(num_samples):
                prediction_value = float(test_predictions[i]) if isinstance(test_predictions[i], np.ndarray) else float(
                    test_predictions[i])
                true_value = float(y_test[i]) if isinstance(y_test[i], np.ndarray) else float(y_test[i])
                #print(f"Sample {i + 1}: Prediction={prediction_value:.4f}, True={true_value:.4f}")

            # 当发现更好的模型时，保存该模型
            if test_rmse < best_test_rmse:
                best_test_rmse = test_rmse
                if best_model_path:
                    os.remove(best_model_path)  # 删除之前保存的最佳模型
                best_model_path = find_available_filename("../retraining/",label_name)
                torch.save(model.state_dict(), best_model_path)
                print(f"Found a better model with Test RMSE: {test_rmse:.4f}. Saved at {best_model_path}")

    print(f"Best model saved at {best_model_path} with Test RMSE: {best_test_rmse:.4f}")
    return best_test_rmse,best_model_path

def find_available_filename(folder_path,label,type = "Torch"):
    # 初始化模型计数器为1
    model_counter = 1

    model_name = f"retrainedRegression{label}{type}_{model_counter}.pt"
    # 检查文件夹中是否存在以'model'开头的文件名，如果有，递增计数器
    while os.path.exists(os.path.join(folder_path, model_name)):
        model_counter += 1
        model_name = f"retrainedRegression{label}{type}_{model_counter}.pt"

    return model_name


if __name__ == "__main__":

    data_path = "../newDataset/Scenario2.csv"
    label_name = "TIT"
    modelPath = "../model/Torch/CARLA_SUN_TIT_BEST.pth"
    MinMaxScalerPath = "../model/minmax/minmax_scaler_model_best.pkl"
    reTrainTITModel(modelPath,MinMaxScalerPath,data_path,label_name)
