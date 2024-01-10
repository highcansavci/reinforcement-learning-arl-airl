import torch
import numpy as np
import os
import torch.nn.functional as F
import torch.nn as nn


class Policy(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions, name, chkp_dir="models/arl_airl"):
        super().__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.checkpoint_file = os.path.join(chkp_dir, name)

        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
        self.bn1 = nn.BatchNorm1d(self.fc1_dims)
        fc1 = 1 / np.sqrt(self.fc1.weight.data.size()[0])
        torch.nn.init.uniform_(self.fc1.weight.data, -fc1, fc1)
        torch.nn.init.uniform_(self.fc1.bias.data, -fc1, fc1)

        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.bn2 = nn.BatchNorm1d(self.fc2_dims)
        fc2 = 1 / np.sqrt(self.fc2.weight.data.size()[0])
        torch.nn.init.uniform_(self.fc2.weight.data, -fc2, fc2)
        torch.nn.init.uniform_(self.fc2.bias.data, -fc2, fc2)

        fc3 = 3e-3
        self.mu = nn.Linear(self.fc2_dims, self.n_actions)
        torch.nn.init.uniform_(self.mu.weight.data, -fc3, fc3)
        torch.nn.init.uniform_(self.mu.bias.data, -fc3, fc3)

        self.optimizer = torch.optim.Adam(self.parameters(), lr)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state):
        state_value = self.fc1(state)
        state_value = self.bn1(state_value)
        state_value = F.relu(state_value)

        state_value = self.fc2(state_value)
        state_value = self.bn2(state_value)
        state_value = F.relu(state_value)

        state_value = torch.tanh(self.mu(state_value))
        return state_value

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))