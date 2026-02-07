import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F

class SoftQNetwork_SAC(nn.Module):
    def __init__(self, n_inputs = 17 + 6, n_outputs = 1) -> None:
        super().__init__()

        self.linear1 = nn.Linear(n_inputs, 256)
        self.linear2 = nn.Linear(256, 256)
        self.linear3 = nn.Linear(256, n_outputs)
        
        # inizialization of weights in a xavier uniform manner and bias to zero
        gain = torch.nn.init.calculate_gain('relu')
        torch.nn.init.xavier_uniform_(self.linear1.weight, gain)
        torch.nn.init.constant_(self.linear1.bias, 0)
        torch.nn.init.xavier_uniform_(self.linear2.weight, gain)
        torch.nn.init.constant_(self.linear2.bias, 0)
        torch.nn.init.xavier_uniform_(self.linear3.weight, gain)
        torch.nn.init.constant_(self.linear3.bias, 0)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class PolicyNetwork_SAC(nn.Module):
    def __init__(self, n_inputs = 17, n_outputs = 6) -> None:
        super().__init__()

        self.linear1 = nn.Linear(n_inputs, 256)
        self.linear2 = nn.Linear(256, 256)

        self.mean_linear = nn.Linear(256, n_outputs)
        self.log_std_linear = nn.Linear(256, n_outputs) 
        
        # inizialization of weights in a xavier uniform manner and bias to zero
        gain = torch.nn.init.calculate_gain('relu')
        torch.nn.init.xavier_uniform_(self.linear1.weight, gain)
        torch.nn.init.constant_(self.linear1.bias, 0)
        torch.nn.init.xavier_uniform_(self.linear2.weight, gain)
        torch.nn.init.constant_(self.linear2.bias, 0)
        torch.nn.init.xavier_uniform_(self.log_std_linear.weight, gain)
        torch.nn.init.constant_(self.log_std_linear.bias, 0)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, -20, 2)

        return mean, log_std
    
    
class Walker_NN_PPO(nn.Module):
    def __init__(self, n_inputs = 17, n_outputs = 6) -> None:
        super().__init__()

        self.backbone = nn.Sequential(
            nn.Linear(n_inputs, 256) , nn.ReLU(),
            )

        self.actor_FC = nn.Sequential(
            nn.Linear(256, 128) , nn.ReLU(),
            nn.Linear(128, 128) , nn.ReLU(),
        )

        self.alpha_head = nn.Sequential(nn.Linear(128, n_outputs), nn.Softplus())
        self.beta_head  = nn.Sequential(nn.Linear(128, n_outputs), nn.Softplus())

        self.critic = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = self.backbone(x)
        V = self.critic(x)

        x = self.actor_FC(x)
        alpha = self.alpha_head(x) + 1
        beta  = self.beta_head(x)  + 1

        return alpha, beta, V
    
    
class CartPole_NN(nn.Module):
    def __init__(self, n_inputs = 4, n_outputs = 1) -> None:
        super().__init__()

        self.backbone = nn.Sequential(nn.Linear(n_inputs, 32), nn.ReLU())

        self.actor_FC = nn.Sequential(
            nn.Linear(32, 32) , nn.ReLU(),
        )

        self.alpha_head = nn.Sequential(nn.Linear(32, n_outputs), nn.Softplus())
        self.beta_head  = nn.Sequential(nn.Linear(32, n_outputs), nn.Softplus())

        self.critic = nn.Sequential(
            nn.Linear(32, 1), nn.ReLU(),
        )

    def forward(self, x):
        x = self.backbone(x)
        V = self.critic(x)

        x = self.actor_FC(x)
        alpha = self.alpha_head(x) + 1
        beta  = self.beta_head(x)  + 1

        return alpha, beta, V
        
        
class HalfCheetah_NN(nn.Module):
    def __init__(self, n_inputs = 17, n_outputs = 6) -> None:
        super().__init__()

        self.backbone = nn.Sequential(
            nn.Linear(n_inputs, 256) , nn.ReLU(),
            )

        self.actor_FC = nn.Sequential(
            nn.Linear(256, 128) , nn.ReLU(),
            nn.Linear(128, 128) , nn.ReLU(),
        )

        self.alpha_head = nn.Sequential(nn.Linear(128, n_outputs), nn.Softplus())
        self.beta_head  = nn.Sequential(nn.Linear(128, n_outputs), nn.Softplus())

        self.critic = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = self.backbone(x)
        V = self.critic(x)

        x = self.actor_FC(x)
        alpha = self.alpha_head(x) + 1
        beta  = self.beta_head(x)  + 1

        return alpha, beta, V
    
class Hopper_NN_PPO(nn.Module):
    def __init__(self, n_inputs = 11, n_outputs = 3) -> None:
        super().__init__()

        self.backbone = nn.Sequential(
            nn.Linear(n_inputs, 64) , nn.ReLU(),
            )

        self.actor_FC = nn.Sequential(
            nn.Linear(64, 64) , nn.ReLU(),
        )

        self.alpha_head = nn.Sequential(nn.Linear(64, n_outputs), nn.Softplus())
        self.beta_head  = nn.Sequential(nn.Linear(64, n_outputs), nn.Softplus())

        self.critic = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        x = self.backbone(x)
        V = self.critic(x)

        x = self.actor_FC(x)
        alpha = self.alpha_head(x) + 1
        beta  = self.beta_head(x)  + 1

        return alpha, beta, V
    
class Swimmer_NN_PPO(nn.Module):
    def __init__(self, n_inputs = 8, n_outputs = 2) -> None:
        super().__init__()

        self.backbone = nn.Sequential(
            nn.Linear(n_inputs, 64) , nn.ReLU(),
            )

        self.actor_FC = nn.Sequential(
            nn.Linear(64, 64) , nn.ReLU(),
        )

        self.alpha_head = nn.Sequential(nn.Linear(64, n_outputs), nn.Softplus())
        self.beta_head  = nn.Sequential(nn.Linear(64, n_outputs), nn.Softplus())

        self.critic = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        x = self.backbone(x)
        V = self.critic(x)

        x = self.actor_FC(x)
        alpha = self.alpha_head(x) + 1
        beta  = self.beta_head(x)  + 1

        return alpha, beta, V