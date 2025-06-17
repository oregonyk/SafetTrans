import argparse
import os
import math
import numpy as np
from evaluate_nocrash_withmodel.utils.data_processing import preprocess_data, readcsv
from sklearn.ensemble import GradientBoostingRegressor  # 导入GradientBoostingRegressor
from sklearn.metrics import mean_squared_error


def train(args):
    # Load and preprocess the data
    X, y = readcsv(args.data_path, args.label_name)
    X_train, X_test, y_train, y_test = preprocess_data(X, y)

    # 初始化GradientBoostingRegressor，可以根据需要调整参数
    model = GradientBoostingRegressor(
        n_estimators=800,  # 迭代次数，可以根据需要调整
        learning_rate=0.01,  # 学习率，可以根据需要调整
        max_depth=13,  # 决策树的深度，可以根据需要调整
        random_state=42,  # 随机种子，用于复现结果
        loss='huber'
    )

    # 训练模型
    model.fit(X_train, y_train)

    # 在训练集上进行预测
    train_predictions = model.predict(X_train)
    train_rmse = math.sqrt(mean_squared_error(y_train, train_predictions))

    # 在测试集上进行预测
    test_predictions = model.predict(X_test)
    test_rmse = math.sqrt(mean_squared_error(y_test, test_predictions))

    print(f"Train RMSE: {train_rmse:.4f}")
    print(f"Test RMSE: {test_rmse:.4f}")

    # 保存模型，如果需要的话
    model_counter = 1  # 初始化计数器为1
    model_path = f"../model/GB/" + os.path.splitext(os.path.basename(args.data_path))[0] + '_' + str(
        model_counter) + ".pkl"
    while os.path.exists(model_path):
        model_counter += 1
        model_path = "../model/GB/" + os.path.splitext(os.path.basename(args.data_path))[0] + str(model_counter) + ".pkl"

    # 保存模型到文件
    import joblib
    joblib.dump(model, model_path)
    print(f"Trained model saved at {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="../dataset/regression/collision/CARLA_collision.csv",
                        help="Path to the data file")
    parser.add_argument("--label_name", type=str, default="SaftyDegree", help="Label name to be excepted")
    args = parser.parse_args()
    if (not os.path.exists(args.data_path)):
        raise RuntimeError("Incorrect data_path")
    train(args)
# Train RMSE: 0.2005
# Test RMSE: 0.8210
# Trained model saved at ../model/GB/CARLA_SUN_collision_1.pkl
# Train RMSE: 0.2101
# Test RMSE: 0.8294
# Trained model saved at ../model/GB/CARLA_SUN_collision2.pkl
# Train RMSE: 0.2233
# Test RMSE: 0.8606
# Trained model saved at ../model/GB/CARLA_SUN_collision3.pkl
# Train RMSE: 0.2222
# Test RMSE: 0.6074
# Trained model saved at ../model/GB/CARLA_collision_1.pkl