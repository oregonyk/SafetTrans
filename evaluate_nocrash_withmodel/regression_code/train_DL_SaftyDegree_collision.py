import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from evaluate_nocrash_withmodel.utils.data_processing import preprocess_data, readcsv
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from RegressionModel import RegressionModelTIT, RegressionModelTET, RegressionModelSaftyDegree


def train(data_path, label_name):
    # Load and preprocess the data
    X, y = readcsv(data_path, label_name)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Convert data to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    # Initialize the model
    model = RegressionModelSaftyDegree(input_dim=X_train.shape[1])

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    num_epochs = 1000
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
            print(f"Epoch [{epoch + 1}/{num_epochs}], Train RMSE: {train_rmse:.4f}, Test RMSE: {test_rmse:.4f}")

            # Output some predictions and true values
            num_samples = 2
            print("Sample predictions and true values:")
            for i in range(num_samples):
                prediction_value = float(test_predictions[i]) if isinstance(test_predictions[i], np.ndarray) else float(
                    test_predictions[i])
                true_value = float(y_test[i]) if isinstance(y_test[i], np.ndarray) else float(y_test[i])
                print(f"Sample {i + 1}: Prediction={prediction_value:.4f}, True={true_value:.4f}")

            # 当发现更好的模型时，保存该模型
            if test_rmse < best_test_rmse:
                best_test_rmse = test_rmse
                if best_model_path:
                    os.remove(best_model_path)  # 删除之前保存的最佳模型
                best_model_path = generate_model_name(data_path, "../model/Torch/",label_name)
                torch.save(model.state_dict(), best_model_path)
                print(f"Found a better model with Test RMSE: {test_rmse:.4f}. Saved at {best_model_path}")

    print(f"Best model saved at {best_model_path} with Test RMSE: {best_test_rmse:.4f}")


def generate_model_name(data_path, folder_path, label):
    filename = os.path.splitext(os.path.basename(data_path))[0]
    model_counter = 1
    model_name = f"Collision{filename}_{label}_{model_counter}.pth"

    while os.path.exists(os.path.join(folder_path, model_name)):
        model_counter += 1
        model_name = f"Collision{filename}_{label}_{model_counter}.pth"

    return os.path.join(folder_path, model_name)

def readcsv(path,label):
    file_path = path
    data = pd.read_csv(file_path)  # 从 data.csv 文件中读取数据
    label_pos = {"SaftyDegree": 13, "TIT": 15, "TET": 16, "Dece": 17}
    #label = "TIT"  # 假设你想要过滤 SaftyDegree 列的特定值
    exclude_value = 0
    # 选择不等于指定值的行
    filtered_data = data[data[label] < exclude_value]
    X = filtered_data.iloc[:, :12].values  # 选择前12列作为特征数据
    label_index = label_pos[label]
    y = filtered_data.iloc[:, label_index].values  # 选择第14列作为目标变量
    # X = data.iloc[:, :12].values  # 选择前12列作为特征数据
    # label_index = label_pos[label]
    # y = data.iloc[:, label_index].values  # 选择第14列作为目标变量
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    print(y)
    MinMaxScalerPath = find_available_filename("../model/Torch/",label,"MinMaxScaler")
    joblib.dump(scaler, MinMaxScalerPath)
    return X_scaled, y
def find_available_filename(folder_path,label,type = "Torch"):
    # 初始化模型计数器为1
    model_counter = 1

    model_name = f"retrainedRegressionCollision{label}{type}_{model_counter}.pt"
    # 检查文件夹中是否存在以'model'开头的文件名，如果有，递增计数器
    while os.path.exists(os.path.join(folder_path, model_name)):
        model_counter += 1
        model_name = f"retrainedRegressionCollision{label}{type}_{model_counter}.pt"

    return model_name

if __name__ == "__main__":

    label_name = "SaftyDegree"
    data_path = "../dataset/dataset_safevar/CARLA_SUN.csv"
    train(data_path, label_name)

