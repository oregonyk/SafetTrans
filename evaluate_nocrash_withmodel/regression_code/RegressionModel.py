import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from evaluate_nocrash_withmodel.binary_code.binary_model import BinaryClassificationModel
from evaluate_nocrash_withmodel.utils.data_processing import preprocess_data, readcsv
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
class RegressionModelTIT(nn.Module):
    def __init__(self, input_dim):
        super(RegressionModelTIT, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.2)
        self.fc4 = nn.Linear(64, 32)
        self.output = nn.Linear(32, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.dropout2(x)
        x = self.fc4(x)
        x = self.relu(x)
        x = self.output(x)
        return x

class RegressionModelTET(nn.Module):
    def __init__(self, input_dim):
        super(RegressionModelTET, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.2)
        self.fc4 = nn.Linear(64, 32)
        self.output = nn.Linear(32, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.dropout2(x)
        x = self.fc4(x)
        x = self.relu(x)
        x = self.output(x)
        return x


class BinaryModelSaftyDegree(nn.Module):
    def __init__(self, input_dim):
        super(BinaryModelSaftyDegree, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.2)
        self.fc4 = nn.Linear(64, 32)
        self.output = nn.Linear(32, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.dropout2(x)
        x = self.fc4(x)
        x = self.relu(x)
        x = self.output(x)
        return x

class RegressionModelSaftyDegree(nn.Module):
    def __init__(self, input_dim):
        super(RegressionModelSaftyDegree, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(0.25)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.dropout3 = nn.Dropout(0.2)
        self.fc5 = nn.Linear(64, 32)
        self.output = nn.Linear(32, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        x = self.relu(x)

        x = self.fc4(x)
        x = self.relu(x)
        x = self.dropout3(x)
        x = self.fc5(x)
        x = self.relu(x)
        x = self.output(x)
        return x