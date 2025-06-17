import argparse
import os
import joblib

import numpy as np
from evaluate_nocrash_withmodel.utils.data_processing import read_testdata
from sklearn.ensemble import GradientBoostingRegressor  # 导入GradientBoostingRegressor

def test(args):
    # Load and preprocess the data
    X, X_origin = read_testdata(args.data_path)

    # 初始化GradientBoostingRegressor，可以根据需要调整参数，确保参数与训练时的模型一致
    model = GradientBoostingRegressor(
        n_estimators=650,  # 迭代次数，保持与训练时一致
        learning_rate=0.1,  # 学习率，保持与训练时一致
        max_depth=13,  # 决策树深度，保持与训练时一致
        random_state=42,  # 随机种子，保持与训练时一致
        loss='huber'
    )

    model_path = args.model_path  # 训练好的Gradient Boosting模型的路径
    if not os.path.exists(model_path):
        raise RuntimeError("Incorrect model_path")

    model = joblib.load(model_path)
    # 使用模型进行预测
    predictions = model.predict(X)  # 预测测试数据

    print(predictions)

    with open('../testing_data/result/predictions.txt', 'w') as f:
        for prediction in predictions:
            f.write(f"{prediction:.4f}\n")

def getPredict(x):

    # Load and preprocess the data
    #X, X_origin = read_testdata(args.data_path)
    x =x.reshape(1, -1)
    # 初始化GradientBoostingRegressor，可以根据需要调整参数，确保参数与训练时的模型一致
    model = GradientBoostingRegressor(
        n_estimators=650,  # 迭代次数，保持与训练时一致
        learning_rate=0.01,  # 学习率，保持与训练时一致
        max_depth=13,  # 决策树深度，保持与训练时一致
        random_state=42,  # 随机种子，保持与训练时一致
        loss='huber'
    )

    model_path = "../../model/GB/CARLA_collision7.pkl"  # 训练好的Gradient Boosting模型的路径
    model = joblib.load(model_path)
    # 使用模型进行预测
    predictions = model.predict(x)  # 预测测试数据
    #print(predictions[0])
    return predictions[0]
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="../model/GB/CARLA_collision5.pkl", help="Path to the trained model")
    parser.add_argument("--data_path", type=str, default="../testing_data/under_prediction/CARLA_collision.csv", help="Path to the data file")
    args = parser.parse_args()

    test(args)
