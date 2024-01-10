import torch
import numpy as np
import os
import torch.nn as nn
import torch.nn.functional as F


class Value(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions, name, chkp_dir="models/arl_airl"):
        """
        Normally the critic should concatenate state and action values; however,
        it makes sense to element-wise add state and action values because of the complexity
        of the state and action values. To do that firstly we need to learn fixed size 1D
        representation of state and action values respectively.
        """
        super().__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.checkpoint_file = os.path.join(chkp_dir, name)

        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
        fc1 = 1 / np.sqrt(self.fc1.weight.data.size()[0])
        torch.nn.init.uniform_(self.fc1.weight.data, -fc1, fc1)
        torch.nn.init.uniform_(self.fc1.bias.data, -fc1, fc1)
        self.bn1 = nn.BatchNorm1d(self.fc1_dims)

        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        fc2 = 1 / np.sqrt(self.fc2.weight.data.size()[0])
        torch.nn.init.uniform_(self.fc2.weight.data, -fc2, fc2)
        torch.nn.init.uniform_(self.fc2.bias.data, -fc2, fc2)
        self.bn2 = nn.BatchNorm1d(self.fc2_dims)

        self.action_value = nn.Linear(self.n_actions, self.fc2_dims)
        fc3 = 3e-4
        self.q = nn.Linear(self.fc2_dims, 1)
        torch.nn.init.uniform_(self.q.weight.data, -fc3, fc3)
        torch.nn.init.uniform_(self.q.bias.data, -fc3, fc3)

        self.optimizer = torch.optim.Adam(self.parameters(), lr, weight_decay=1e-2)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state, action):
        state_value = self.fc1(state)
        state_value = self.bn1(state_value)
        state_value = F.relu(state_value)

        state_value = self.fc2(state_value)
        state_value = self.bn2(state_value)
        state_value = F.relu(state_value)

        action_value = self.action_value(action)
        state_action_value = F.relu(torch.add(state_value, action_value))
        state_action_value = self.q(state_action_value)

        return state_action_value

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))