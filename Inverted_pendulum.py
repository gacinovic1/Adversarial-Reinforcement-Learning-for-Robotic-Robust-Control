import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Beta

import PPO_RARL as PPO

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

class CartPole(gym.Wrapper):
    def __init__(self, render_mode = None) -> None:
        self.is_norm_wrapper = False
        if render_mode == 'human':
            self.env = gym.make('InvertedPendulum-v5', reset_noise_scale=0.1, render_mode = 'human')
        else:
            self.env = gym.make('InvertedPendulum-v5', reset_noise_scale=0.1)

    def _preprocess_action(self, action) -> float:
        return np.asarray(action.squeeze(0).cpu()) * 6 - 3 # scale action in range [-3, 3]
    
    def _postprocess_state(self, state) -> torch.tensor:
        return torch.tensor(state, dtype=torch.float32).reshape(1,-1)
    
    def step(self, action_player, action_opponent = 0):
        state, reward, terminated, truncated, info = self.env.step(self._preprocess_action(action_player + (action_opponent*0.02 - 0.01)))

        return self._postprocess_state(state), reward, terminated, truncated, info
    
    def step_alone(self, action_player):
        state, reward, terminated, truncated, info = self.env.step(self._preprocess_action(action_player))

        return self._postprocess_state(state), reward, terminated, truncated, info
    
    def reset(self):
        state, info = self.env.reset()
        return self._postprocess_state(state), info
    
    def close(self):
        self.env.close()

def Test(RL, env, steps = 10_000):
    s, _ = env.reset()
    reward = 0
    attempts = 1
    for i in range(steps):
        s, r, term, tronc, _ = env.step_alone(RL.act(s))
        reward += r
        if term or tronc: 
            s, _ = env.reset()
            attempts += 1
        if np.mod(i, 1000) == 0: print('#', end='', flush=True)
    env.close()

    print(f'\n---> rewards: {reward/attempts} | gained in {attempts} attempts')

def Perturbate_env(env, pert = 0):
    print(f"Original pendolum mass: {env.unwrapped.model.body_mass[2]}")

    # modify pendolum mass
    env.unwrapped.model.body_mass[2] += env.unwrapped.model.body_mass[2]*pert
    print(f"New pendolum mass: {env.unwrapped.model.body_mass[2]}")

def main(render = True, train = True, alg = 'RARL', pm_pert = 0):
    # init environment and neural network
    render_mode = 'human'
    env = CartPole(render_mode if render else None)

    player = CartPole_NN()
    opponent = CartPole_NN()

    # init the PPO or RARL_PPO algorithm
    if alg == 'RARL':
        rarl_ppo = PPO.RARL_PPO(player, opponent, env, print_flag=True, name='CartPole_Adverarial_model')
        if train: rarl_ppo.train(player_episode=10, 
                             opponent_episode=4, 
                             episodes=200, 
                             mini_bach=64, 
                             max_steps_rollouts=1024, 
                             continue_prev_train=False)
        rarl_ppo.load()
    elif alg == 'PPO':
        ppo = PPO.PPO(player, env, print_flag=False, name='CartPole_model')
        if train: ppo.train(episodes=200, mini_bach=64, max_steps_rollouts=1024, continue_prev_train=False)
        ppo.load()

    env.close()

    # render the simulation if needed
    if not render: return

    # perturbate the model paramether
    Perturbate_env(env, pm_pert)
    
    # choise the algorithm for run the simulation
    RL = ppo if alg == 'PPO' else rarl_ppo
    Test(RL, env, 10_000)

if __name__ == '__main__':
    main(render=True, train=False, pm_pert = -0.1, alg = 'PPO')