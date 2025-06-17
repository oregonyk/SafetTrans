import joblib
import numpy as np

from evaluate_nocrash_withmodel.binary_code.binary_model import BinaryClassificationModel

import argparse
import os

from evaluate_nocrash_withmodel.utils.data_processing import  preprocess_data, read_testdata
import torch
import torch.nn as nn
import torch.optim as optim
import math
from sklearn.preprocessing import MinMaxScaler
def getBinary(X):
    # Load and preprocess the data
    scaler = MinMaxScaler()
    scaler = joblib.load("../model/minmax/minmax_scaler_model.pkl")
    X = scaler.transform(X.reshape(1, -1))
    model = BinaryClassificationModel(12)
    model.load_state_dict(torch.load("../model/CARLA_SUN_binary_1.pt"))  # 加载训练好的模型
    # 设置为评估模式
    model.eval()
    # Testing loop
    predictions = []
    with torch.no_grad():  # 不计算梯度
        # 将输入数据转换为 PyTorch Tensor
        x_tensor = torch.tensor(X, dtype=torch.float32)
        # 使用模型进行推理
        output = model(x_tensor.unsqueeze(0))  # 添加批次维度
        # 将模型的预测结果添加到列表中
        #print(output.item())  # 如果输出维度是1，可以使用item()来获取值
    return  output.item()
def test(args):
    # Load and preprocess the data
    X,X_original = read_testdata(args.data_path)
    model = BinaryClassificationModel(args.input_dim)
    model.load_state_dict(torch.load(args.model_path))  # 加载训练好的模型

    # 设置为评估模式
    model.eval()
    batch_size = args.batch_size
    total_correct = 0
    # Testing loop
    predictions = []

    with torch.no_grad():  # 不计算梯度
        for x in X:
            print(x)
            # 将输入数据转换为 PyTorch Tensor
            x_tensor = torch.tensor(x, dtype=torch.float32)
            # 使用模型进行推理
            output = model(x_tensor.unsqueeze(0))  # 添加批次维度
            # 将模型的预测结果添加到列表中
            print(output.item())
            predictions.append(output.item())  # 如果输出维度是1，可以使用item()来获取值

    with open('../dataset/testing_data/result/predictions_binary.txt', 'a+') as f:
        for prediction in predictions:
            if(prediction<0.5):
                f.write("0\n")
            else:
                f.write("1\n")
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dim", type=int, default=12, help="Input dimension")
    # parser.add_argument("--model_path", type=str, default="../model/CARLA_SUN_binary_1.pt",
    #                     help="Path to the trained model")
    parser.add_argument("--model_path", type=str, default="../model/retrained_model_1.pt",
                        help="Path to the trained model")
    parser.add_argument("--data_path", type=str, default="../dataset/testing_data/under_prediction/CARLA_binary.csv",
                        help="Path to the data file")
    parser.add_argument("--num_epochs", type=int, default=2, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--label_name", type=str, default="SaftyDegree", help="Label name to be excepted")
    parser.add_argument("--lr", type=float, default=0.001, help="学习率")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="权重衰减")
    parser.add_argument("--output_dim", type=int, default=1, help="输出维度")
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=4)
    args = parser.parse_args()

    if (not os.path.exists(args.model_path)):
        raise RuntimeError("Incorrect model_path")
    test(args)