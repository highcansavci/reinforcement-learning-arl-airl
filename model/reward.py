import torch.nn as nn
import torch
import os


def estimate_reward(states, actions, reward_net, min_reward, max_reward):
    reward_net.eval()
    with torch.no_grad():
        concatenated_input = torch.cat((states, actions), dim=1)
        predicted_rewards = reward_net(concatenated_input)
        return (predicted_rewards - min_reward) * (max_reward - min_reward) / 2 + min_reward


class Reward(nn.Module):
    def __init__(self, input_dim, chkpt_dir="models/arl_airl"):
        super(Reward, self).__init__()
        self.reward = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh()
        )
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        self.checkpoint_file = os.path.join(chkpt_dir, "torch_reward")

    def forward(self, state):
        disc = self.reward(state)
        return disc

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file, map_location=self.device))
