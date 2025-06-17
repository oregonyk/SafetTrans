import argparse
import os
import numpy as np

from binary_model import BinaryClassificationModel
from evaluate_nocrash_withmodel.utils.data_processing import preprocess_data, readcsv_binary
import torch
import torch.nn as nn
import torch.optim as optim
import math
from imblearn.over_sampling import SMOTE


def train(args):
    # Load and preprocess the data
    X, y = readcsv_binary(args.data_path, args.label_name)

    # 使用SMOTE过采样来平衡数据集
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    X_train, X_test, y_train, y_test = preprocess_data(X_resampled, y_resampled)

    model = BinaryClassificationModel(args.input_dim)

    # Define loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_epochs = args.num_epochs
    batch_size = args.batch_size

    losses_tr = []  # 用于存储训练损失

    # 训练模型
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        total_correct = 0

        for i in range(0, len(X_train), batch_size):
            batch_inputs = torch.tensor(X_train[i:i + batch_size], dtype=torch.float32)
            batch_targets = torch.tensor(y_train[i:i + batch_size], dtype=torch.float32).view(-1, 1)  # 调整目标变量的形状

            optimizer.zero_grad()
            outputs = model(batch_inputs)
            loss = criterion(outputs, batch_targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            predictions = (outputs >= 0.5).float()
            total_correct += (predictions == batch_targets).sum().item()

        average_loss = total_loss / (len(X_train) / batch_size)
        accuracy = total_correct / len(X_train)

        print(f"Epoch [{epoch + 1}/{num_epochs}] - Train Loss: {average_loss:.4f} - Train Accuracy: {accuracy:.4f}")

        # 在每个epoch结束后使用X_test进行测试
        model.eval()
        test_loss = 0.0
        test_correct = 0

        with torch.no_grad():
            for i in range(0, len(X_test), batch_size):
                batch_inputs = torch.tensor(X_test[i:i + batch_size], dtype=torch.float32)
                batch_targets = torch.tensor(y_test[i:i + batch_size], dtype=torch.float32).view(-1, 1)

                outputs = model(batch_inputs)
                loss = criterion(outputs, batch_targets)

                test_loss += loss.item()

                predictions = (outputs >= 0.5).float()
                test_correct += (predictions == batch_targets).sum().item()

        test_average_loss = test_loss / (len(X_test) / batch_size)
        test_accuracy = test_correct / len(X_test)

        print(
            f"Epoch [{epoch + 1}/{num_epochs}] - Test Loss: {test_average_loss:.4f} \t- Test Accuracy: {test_accuracy:.4f}")

    model_counter = 1  # 初始化计数器为1
    model_path = f"../model/" + os.path.splitext(os.path.basename(args.data_path))[0] + "_binary_" + str(
        model_counter) + ".pt"
    while os.path.exists(model_path):
        model_counter += 1
        model_path = "../model/" + os.path.splitext(os.path.basename(args.data_path))[0] +"_binary_" + str(model_counter) + ".pt"
    torch.save(model.state_dict(), model_path)
    print(f"Trained model saved at {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="../dataset/binary/CARLA_SUN.csv", help="Path to the data file")
    parser.add_argument("--input_dim", type=int, default=12, help="Input dimension")
    parser.add_argument("--num_epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--label_name", type=str, default="SaftyDegree", help="Label name to be excepted")
    parser.add_argument("--lr", type=float, default=0.001, help="学习率")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="权重衰减")
    parser.add_argument("--output_dim", type=int, default=1, help="输出维度")
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=4)
    args = parser.parse_args()
    if (not os.path.exists(args.data_path)):
        raise RuntimeError("Incorrect data_path")
    train(args)
