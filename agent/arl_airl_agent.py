import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from model.ounoise import OUNoise
from replay_buffer.replay_buffer import ReplayBuffer
from model.reward import Reward, estimate_reward
from model.discriminator import Discriminator
from model.policy import Policy
from model.value import Value


class Agent:
    def __init__(self, input_dim, env, gamma=0.99, tau=5e-3, gae_lambda=0.9, n_actions=2, batch_size=64, n_epochs=10, max_size=int(1e6), fc1_dims=400, fc2_dims=300):
        self.gamma = gamma
        self.tau = tau
        self.min_reward = -500
        self.max_reward = 500
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.min_action = env.action_space.high[0]
        self.max_action = env.action_space.low[0]
        self.gae_lambda = gae_lambda
        self.noise = OUNoise(mu=np.zeros(n_actions))
        self.arl_discriminator = Discriminator(1, chkpt_dir="models/arl")
        self.arl_policy = Policy(lr=3e-3, input_dims=input_dim, fc1_dims=fc1_dims, fc2_dims=fc2_dims, n_actions=n_actions, chkp_dir="models/arl", name="torch_policy")
        self.arl_value = Value(lr=3e-4, input_dims=input_dim, fc1_dims=fc1_dims, fc2_dims=fc2_dims, n_actions=n_actions, chkp_dir="models/arl", name="torch_value")
        self.airl_discriminator = Discriminator(n_actions, chkpt_dir="models/airl")
        self.airl_policy = Policy(lr=3e-3, input_dims=input_dim, fc1_dims=400, fc2_dims=300, n_actions=2, chkp_dir="models/airl", name="torch_policy")
        self.airl_reward = Reward(input_dim + n_actions, chkpt_dir="models/airl")
        self.airl_value = Value(lr=3e-4, input_dims=input_dim, fc1_dims=fc1_dims, fc2_dims=fc2_dims, n_actions=n_actions, chkp_dir="models/airl", name="torch_value")
        self.memory = ReplayBuffer(max_size, input_dim, n_actions)
        self.discriminator_criterion = nn.MSELoss()
        self.update_network_parameters(tau=1)

    def remember(self, state, action, reward, next_state, done):
        self.memory.store_transition(state, action, reward, next_state, done)

    def soft_update(self, target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - tau) + param.data * tau
            )

    def save_models(self):
        print("Saving Models...")
        self.arl_discriminator.save_checkpoint()
        self.arl_policy.save_checkpoint()
        self.arl_value.save_checkpoint()
        self.airl_discriminator.save_checkpoint()
        self.airl_reward.save_checkpoint()
        self.airl_policy.save_checkpoint()
        self.airl_value.save_checkpoint()

    def load_models(self):
        print("Loading Models...")
        self.arl_discriminator.load_checkpoint()
        self.arl_policy.load_checkpoint()
        self.arl_value.load_checkpoint()
        self.airl_discriminator.load_checkpoint()
        self.airl_reward.load_checkpoint()
        self.airl_policy.load_checkpoint()
        self.airl_value.load_checkpoint()

    def choose_action(self, observation):
        self.airl_policy.eval()
        state = torch.tensor(np.array([observation]), device=self.arl_discriminator.device, dtype=torch.float32)
        mu = self.airl_policy(state).to(self.arl_discriminator.device)
        mu = (mu - self.min_action) * (self.max_action - self.min_action) / 2 + self.min_action
        mu_prime = mu + torch.tensor(self.noise(), dtype=torch.float32, device=self.arl_discriminator.device)
        mu_prime = torch.clamp(mu_prime, self.min_action, self.max_action)
        self.airl_policy.train()
        return mu_prime.detach().cpu().numpy()

    def learn(self):
        if self.memory.mem_ctr < self.batch_size:
            return
        states, actions, rewards, next_states, dones = self.memory.sample_buffer(self.batch_size)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.arl_policy.device)
        next_states = torch.tensor(next_states, dtype=torch.float32, device=self.arl_policy.device)
        actions = torch.tensor(actions, dtype=torch.float32, device=self.arl_policy.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.arl_policy.device)
        states = torch.tensor(states, dtype=torch.float32, device=self.arl_policy.device)

        # Train ARL DDPG
        self.arl_value.eval()
        self.airl_value.eval()
        self.airl_policy.eval()

        target_actions = self.airl_policy(next_states)
        target_actions = (target_actions - self.min_action) * (self.max_action - self.min_action) / 2 + self.min_action
        critic_target_value = self.airl_value(next_states, target_actions)
        critic_value = self.airl_value(states, actions)

        target = rewards.reshape(self.batch_size, 1) + self.gamma * critic_target_value * dones.reshape(self.batch_size, 1)

        self.arl_value.train()
        self.arl_value.optimizer.zero_grad()
        critic_loss = F.mse_loss(target, critic_value)
        critic_loss.backward()
        self.arl_value.optimizer.step()

        self.arl_value.eval()
        self.arl_policy.optimizer.zero_grad()
        mu = self.arl_policy(states)
        mu = (mu - self.min_action) * (self.max_action - self.min_action) / 2 + self.min_action
        self.arl_policy.train()
        actor_loss = -self.arl_value(states, mu).mean()
        actor_loss.backward()
        self.arl_policy.optimizer.step()

        # Train Discriminator AIRL
        true_policy_labels = self.airl_discriminator(actions)
        false_policy_labels = self.arl_policy(states).to(self.arl_discriminator.device)
        false_policy_labels = (false_policy_labels - self.min_action) * (
                    self.max_action - self.min_action) / 2 + self.min_action
        false_policy_labels = self.airl_discriminator(false_policy_labels)
        discriminator_airl_loss = self.discriminator_criterion(true_policy_labels, false_policy_labels)
        self.airl_discriminator.optimizer.zero_grad()
        discriminator_airl_loss.backward()
        self.airl_discriminator.optimizer.step()

        # Train AIRL
        concatenated_input = torch.cat((states, actions), dim=1)
        predicted_rewards = self.airl_reward(concatenated_input)
        false_reward_labels = (predicted_rewards - self.min_reward) * (self.max_reward - self.min_reward) / 2 + self.min_reward
        true_reward_labels = self.arl_discriminator(rewards.reshape(self.batch_size, 1))
        false_reward_labels = self.arl_discriminator(false_reward_labels)
        discriminator_arl_loss = self.discriminator_criterion(true_reward_labels, false_reward_labels)

        self.arl_discriminator.optimizer.zero_grad()
        self.airl_reward.optimizer.zero_grad()
        discriminator_arl_loss.backward()
        self.airl_reward.optimizer.step()
        self.arl_discriminator.optimizer.step()
        self.update_network_parameters()

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        self.soft_update(self.airl_policy, self.arl_policy, tau)
        self.soft_update(self.airl_value, self.arl_value, tau)