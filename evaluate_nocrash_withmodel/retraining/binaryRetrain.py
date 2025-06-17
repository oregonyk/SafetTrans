from imblearn.over_sampling import SMOTE
from pymoo.core.problem import ElementwiseProblem
import torch
import torch.nn as nn
import torch.optim as optim
import math

from sklearn.preprocessing import MinMaxScaler

from evaluate_nocrash_withmodel.binary_code.binary_model import BinaryClassificationModel
from evaluate_nocrash_withmodel.utils.data_processing import readcsv, readcsv_binary, preprocess_data

from runners import NoCrashEvalRunner


def processBinaryData(path):
    X_scaled, y = readcsv_binary(path, "SaftyDegree")
    return X_scaled, y


import os

# Define a global counter for the models
model_counter = 1
def find_available_filename(folder_path):
    # 初始化模型计数器为1
    model_counter = 1
    model_name = f"retrainedBinaryModel_{model_counter}.pt"

    # 检查文件夹中是否存在以'model'开头的文件名，如果有，递增计数器
    while os.path.exists(os.path.join(folder_path, model_name)):
        model_counter += 1
        model_name = f"retrainedBinaryModel_{model_counter}.pt"

    return model_name


def reTrainBinaryModel(X_scaled, y, modelPath):
    global model_counter
    modelSaveFolderPath = "../model/"
    unique_filename = find_available_filename(modelSaveFolderPath)
    # smote = SMOTE(random_state=42)
    # X_resampled, y_resampled = smote.fit_resample(X_scaled, y)
    # X, y = X_resampled, y_resampled
    X, y = X_scaled, y
    # Load the pre-trained model
    model = BinaryClassificationModel(12)
    model.load_state_dict(torch.load(modelPath))

    # Freeze the early layers to retain pre-trained weights
    for param in model.parameters():
        param.requires_grad = False

    # Modify the last layer for binary classification
    num_features = model.fc2.in_features  # Assuming 'fc2' is the last fully connected layer
    model.fc2 = nn.Linear(num_features, 1)  # Replace the last layer with a single output neuron
    model.sigmoid = nn.Sigmoid()  # Sigmoid activation for binary classification

    # Define loss function and optimizer
    criterion = nn.BCELoss()  # Binary Cross Entropy Loss for binary classification
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the modified model
    num_epochs = 50
    batch_size = 64
    train_data = torch.utils.data.TensorDataset(torch.tensor(X, dtype=torch.float32),
                                                torch.tensor(y, dtype=torch.float32))
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

    # Train the model and calculate RMSE
    running_loss = 0.0
    for epoch in range(num_epochs):
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)  # Squeeze to match the shape of labels
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            print(loss.item())
    # Calculate RMSE
    rmse = math.sqrt(running_loss / (len(train_loader) * num_epochs))
    #print(rmse)
    # Save the model with a numbered filename

    model_save_path = os.path.join(modelSaveFolderPath, unique_filename)
    torch.save(model.state_dict(), model_save_path)

    # Increment the model counter for the next model
    model_counter += 1

    # Return the final RMSE and the path to the saved model
    return rmse, model_save_path


if __name__ =="__main__":
    writtenPath = "/home/yko/mjw/WorldOnRails-release/evaluate_nocrash_withmodel/newDataset/modelWithReal.csv"
    binaryModelPath = '../model/CARLA_SUN_binary_2.pt'
    X_scaled, y = processBinaryData(writtenPath)
    #print(X_scaled,y)
    accuracy = reTrainBinaryModel(X_scaled, y,binaryModelPath)
    print(accuracy)