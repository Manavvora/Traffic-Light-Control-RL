import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys
import os


class TrainModel(nn.Module):
    def __init__(self, num_layers, width, input_dim, output_dim):
        super(TrainModel, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.fc1 = nn.Linear(input_dim, width)
        self.hidden_layers = nn.ModuleList([nn.Linear(width, width) for _ in range(num_layers)])
        self.fc2 = nn.Linear(width, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        for layer in self.hidden_layers:
            x = layer(x)
            x = self.relu(x)
        x = self.fc2(x)
        return x


class TrainModel:
    def __init__(self, num_layers, width, batch_size, learning_rate, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.model = TrainModel(num_layers, width, input_dim, output_dim)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

    def predict_one(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        with torch.no_grad():
            return self.model(state).numpy()

    def predict_batch(self, states):
        states = torch.from_numpy(states).float()
        with torch.no_grad():
            return self.model(states).numpy()

    def train_batch(self, states, q_sa):
        self.model.train()
        states = torch.from_numpy(states).float()
        q_sa = torch.from_numpy(q_sa).float()
        self.optimizer.zero_grad()
        predictions = self.model(states)
        loss = self.criterion(predictions, q_sa)
        loss.backward()
        self.optimizer.step()

    def save_model(self, path):
        torch.save(self.model.state_dict(), os.path.join(path, 'trained_model.pth'))
        dummy_input = torch.randn(1, self.input_dim)
        traced_model = torch.jit.trace(self.model, dummy_input)
        traced_model.save(os.path.join(path, 'trained_model.pt'))

    @classmethod
    def load_model(cls, path, num_layers, width, input_dim, output_dim):
        model = cls(num_layers, width, 1, 1, input_dim, output_dim)
        model.model.load_state_dict(torch.load(os.path.join(path, 'trained_model.pth')))
        return model

    @property
    def input_dim(self):
        return self._input_dim


    @property
    def output_dim(self):
        return self._output_dim


    @property
    def batch_size(self):
        return self._batch_size


class TestModel:
    def __init__(self, input_dim, model_path):
        self.input_dim = input_dim
        self.model = torch.jit.load(os.path.join(model_path, 'trained_model.pt'))

    def predict_one(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        with torch.no_grad():
            return self.model(state).numpy()

    @property
    def input_dim(self):
        return self._input_dim