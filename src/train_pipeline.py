import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

from src.config import config
import src.preprocessing.preprocessors as pp
from src.preprocessing.data_management import load_dataset, save_model, load_model
import src.trained_models.pipeline as pl

class CustomNet(nn.Module):
    def __init__(self):
        super(CustomNet, self).__init__()
        self.fc1 = nn.Linear(2, config.NUM_LAYERS)  # Input layer to first hidden layer

        self.layers = nn.ModuleList()
        for l in range(1, config.NUM_LAYERS):
            if config.f[l] == "sigmoid":
                self.layers.append(nn.Sigmoid())
            elif config.f[l] == "tanh":
                self.layers.append(nn.Tanh())
            elif config.f[l] == "relu":
                self.layers.append(nn.ReLU())
            # Hidden layers
            self.layers.append(nn.Linear(config.NUM_LAYERS, config.NUM_LAYERS))

    def forward(self, x):
        x = self.fc1(x)
        for layer in self.layers:
            x = layer(x)
        return x

def run_training(tol, epsilon, mini_batch_size=2):
    epoch_counter = 0
    mse = 1
    loss_per_epoch = []

    training_data = load_dataset("train.csv")

    obj = pp.preprocess_data()
    obj.fit(training_data.iloc[:, 0:2], training_data.iloc[:, 2])
    X_train, Y_train = obj.transform(training_data.iloc[:, 0:2], training_data.iloc[:, 2])

    model = CustomNet()
    optimizer = optim.SGD(model.parameters(), lr=epsilon)

    num_batches = X_train.shape[0] // mini_batch_size

    while epoch_counter < 2:  # Change the condition to run for 1000 epochs
        mse = 0

        for batch in range(num_batches):
            batch_X = torch.tensor(X_train[batch * mini_batch_size:(batch + 1) * mini_batch_size], dtype=torch.float32)
            batch_Y = torch.tensor(Y_train[batch * mini_batch_size:(batch + 1) * mini_batch_size], dtype=torch.float32)

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = nn.MSELoss()(outputs, batch_Y)
            loss.backward()
            optimizer.step()

            mse += loss.item()

        mse /= X_train.shape[0]
        epoch_counter += 1
        loss_per_epoch.append(mse)
        print("Epoch # {}, Loss = {}".format(epoch_counter, mse))

        if epoch_counter > 1 and abs(loss_per_epoch[epoch_counter - 1] - loss_per_epoch[epoch_counter - 2]) < tol:
            break

        if epoch_counter == 1:
            continue  # Skip the first check because loss_per_epoch has only one element initially

    return loss_per_epoch

if __name__ == "__main__":
    run_training(10**(-8), 10**(-7), mini_batch_size=2)
    # Save your model using appropriate PyTorch mechanisms
