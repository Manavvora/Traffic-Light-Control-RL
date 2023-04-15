import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import sys
from utils import import_train_configuration

config = import_train_configuration(config_file='training_settings.ini')

class TrainModel(nn.Module):
    def __init__(self, num_layers, width, batch_size, learning_rate, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.model = self._build_model(num_layers, width)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def _build_model(self, num_layers, width):
        layers = []
        layers.append(nn.Linear(self.input_dim, width))
        layers.append(nn.ReLU())
        for _ in range(num_layers):
            layers.append(nn.Linear(width, width))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(width, self.output_dim))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.model(x)
        return x 

    def predict_one(self, state):
        state = np.reshape(state, [1, self.input_dim])
        state = torch.tensor(state, dtype=torch.float32)
        with torch.no_grad():
            return self.model(state).numpy()

    def predict_batch(self, states):
        states = torch.tensor(states, dtype=torch.float32)
        with torch.no_grad():
            return self.model(states).numpy()

    def train_batch(self, states, q_sa):
        states = torch.tensor(states, dtype=torch.float32)
        q_sa = torch.tensor(q_sa, dtype=torch.float32)
        self.optimizer.zero_grad()
        outputs = self.model(states)
        loss = self.criterion(outputs, q_sa)
        loss.backward()
        self.optimizer.step()

    def save_model(self, path):
        torch.save(self.model.state_dict(), os.path.join(path, 'trained_model.pt'))

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
        self.model.eval()


class TestModel:
    def __init__(self, input_dim, model_path):
        self.input_dim = input_dim
        self.model = self._load_my_model(model_path)


    def _load_my_model(self, model_folder_path):
        model_file_path = os.path.join(model_folder_path, 'trained_model.pt')
        
        if os.path.isfile(model_file_path):
            loaded_model_state_dict = torch.load(model_file_path)
            model = TrainModel(input_dim=self.input_dim, output_dim=config['num_actions'], batch_size=config['batch_size'], learning_rate=config['learning_rate'], num_layers=config['num_layers'], width=config['width_layers'])
            loaded_model_state_dict = {f"model.{k}": v for k, v in loaded_model_state_dict.items()}
            model.load_state_dict(loaded_model_state_dict)
            model.eval()
            return model
        else:
            sys.exit("Model number not found")

    def predict_one(self, state):
        state = np.reshape(state, [1, self.input_dim])
        state = torch.tensor(state, dtype=torch.float32)
        with torch.no_grad():
            return self.model(state).numpy()
