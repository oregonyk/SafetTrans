import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def preprocess_data(X, y, test_size=0.2, random_state=None):
    # 标准化数据
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test


def readcsv(path,label):
    '''file_path=path
    data = pd.read_csv(file_path)  # 假设数据保存在data.csv文件中
    label_pos={"SaftyDegree":13,"TIT":15,"TET":16,"Dece":17}
    X = data.iloc[:, :12].values  # 选择前12列作为特征数据
    label_index=label_pos[label]
    y = data.iloc[:, label_index].values  # 选择第14列作为目标变量
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    #print(X,X_scaled)
    return X_scaled,y'''
    file_path = path
    data = pd.read_csv(file_path)  # 从 data.csv 文件中读取数据
    label_pos = {"SaftyDegree": 13, "TIT": 15, "TET": 16, "Dece": 17}
    label = "SaftyDegree"  # 假设你想要过滤 SaftyDegree 列的特定值
    exclude_value = 11.549
    # 选择不等于指定值的行
    filtered_data = data[data[label] != exclude_value]
    X = filtered_data.iloc[:, :12].values  # 选择前12列作为特征数据
    label_index = label_pos[label]
    y = filtered_data.iloc[:, label_index].values  # 选择第14列作为目标变量
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y


def read_testdata(path):
    file_path=path
    data = pd.read_csv(file_path)  # 假设数据保存在data.csv文件中
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(data)
    print(data,X_scaled)
    return X_scaled,data

def readcsv_binary(path,label):
    file_path = path
    data = pd.read_csv(file_path)  # 假设数据保存在data.csv文件中
    label_pos = {"SaftyDegree": 13, "TIT": 15, "TET": 16, "Dece": 17}
    X = data.iloc[:, :12].values  # 选择前12列作为特征数据
    label_index = label_pos[label]
    y = data.iloc[:, label_index].values  # 选择第14列作为目标变量
    #print(y)
    # 将小于0的目标变量改为-1，大于等于0的改为1
    y = [0 if val < 0 else 1 for val in y]
    #print(y)
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    scaler_filename = "../model/minmax/minmax_scaler_model.pkl"
    joblib.dump(scaler, scaler_filename)
    return X_scaled, y
