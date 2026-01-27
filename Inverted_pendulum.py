import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Beta

import PPO_RARL as PPO
import SAC_RARL as SAC 
import ENV_Wrapper as Env
import architectures as net

import csv



class CartPole(gym.Wrapper):
    def __init__(self, render_mode = None, algorithm = 'SAC') -> None:
        self.is_norm_wrapper = False
        self.algorithm = algorithm
        
        if render_mode == 'human':
            self.env = gym.make('InvertedPendulum-v5', reset_noise_scale=0.1, render_mode = 'human')
        else:
            self.env = gym.make('InvertedPendulum-v5', reset_noise_scale=0.1)

    def _preprocess_action(self, action) -> float:
        if self.algorithm in  ['PPO', 'PPO_RARL', 'RARL']:
            return np.asarray(action.squeeze(0).cpu()) * 6 - 3 # scale action in range [-3, 3]
        if self.algorithm == 'SAC' or self.algorithm == 'RARL_SAC':
            return action.detach().cpu().numpy()[0]* 3 # scale action in range [-3, 3]
        
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

def Test(RL, env, steps = 10_000) -> tuple[float, int, int, list[float, int]]:
    
    env.update_running_stats = False
    s, _ = env.reset()
    rew_list = [(0, 0)]
    reward = 0
    attempts = 0
    for i in range(steps):
        env.update_running_stats = False 
        s, r, term, tronc, _ = env.step_alone(RL.act(s))
        reward += r
        if term or tronc: 
            s, _ = env.reset()  
            attempts += 1
            rew_list.append((reward - np.sum([r[0] for r in rew_list]), i+1 - np.sum([s[1] for s in rew_list])))
        if np.mod(i, 1000) == 0: print('#', end='', flush=True)
    env.close()

    print(f'\n---> rewards: {reward/attempts} | gained in {attempts} attempts')

    return (reward, attempts, steps, rew_list[1:])

def Perturbate_env(env, pert = 0):
    print(f"Original pendolum mass: {env.unwrapped.model.body_mass[2]}")

    # modify pendolum mass
    env.unwrapped.model.body_mass[2] += env.unwrapped.model.body_mass[2]*pert
    print(f"New pendolum mass: {env.unwrapped.model.body_mass[2]}")

def main(render = True, train = True, alg = 'RARL', pm_pert = 0, model_to_load = ''):
    # init environment and neural network
    render_mode = ''#'human'
    env = CartPole(render_mode if render else None, alg)

    if alg in ['PPO', 'RARL_PPO', 'RARL']:
        
        player = net.CartPole_NN()
        opponent = net.CartPole_NN() # 2 output for X, Y forces on both feat
    
    if alg in ['SAC', 'RARL_SAC']:
        
        player = {
            'policy': net.PolicyNetwork_SAC(),
            'Q1_target': net.SoftQNetwork_SAC(),
            'Q2_target': net.SoftQNetwork_SAC(),
            'Q1'    : net.SoftQNetwork_SAC(),
            'Q2'    : net.SoftQNetwork_SAC()
        }
        opponent = {
            'policy': net.PolicyNetwork_SAC(),
            'Q1_target': net.SoftQNetwork_SAC(),
            'Q2_target': net.SoftQNetwork_SAC(),
            'Q1'    : net.SoftQNetwork_SAC(),
            'Q2'    : net.SoftQNetwork_SAC()
        }

    # init the PPO or RARL_PPO algorithm
    if alg == 'RARL_PPO':
        rarl_ppo = PPO.RARL_PPO(player, opponent, env, print_flag=True, name='CartPole_Adverarial_model' if model_to_load == '' else model_to_load)
        if train: rarl_ppo.train(player_episode=10, 
                             opponent_episode=4, 
                             episodes=200, 
                             mini_bach=64, 
                             max_steps_rollouts=1024, 
                             continue_prev_train=False)
        rarl_ppo.load()
    elif alg == 'PPO':
        ppo = PPO.PPO(player, env, print_flag=False, name='CartPole_model' if model_to_load == '' else model_to_load)
        if train: ppo.train(episodes=200, mini_bach=64, max_steps_rollouts=1024, continue_prev_train=False)
        ppo.load()
    
    if alg == 'RARL_SAC':
        rarl_sac = SAC.RARL_SAC(player, opponent, env, print_flag=False, lr_Q=3e-4, lr_player=1e-4, name='Inverted_Pendulum_Adversarial_SAC_model')
        if train: rarl_sac.train(player_episode=10, 
                             opponent_episode=4, 
                             episodes=1000, 
                             epoch = 1,
                             mini_bach=128, 
                             max_steps_rollouts=1024, 
                             continue_prev_train=False)
        rarl_sac.load()
        
    elif alg == 'SAC':
        sac = SAC.SAC(player['Q1_target'], player['Q2_target'], player['Q1'], player ['Q2'], player['policy'], env, print_flag=False, lr_Q=3e-4, lr_pi=1e-4, name='Inverted_Pendulum_model_SAC')
        if train: sac.train(episodes=1000, epoch=1, mini_batch=128, max_steps_rollouts=1024, continue_prev_train=False)
        sac.load()

    env.close()

    # render the simulation if needed
    if not render: return

    # perturbate the model paramether
    Perturbate_env(env, pm_pert)

    # choise the algorithm for run the simulation
    RL = ppo  if alg == 'PPO' else \
        rarl_ppo if alg == 'RARL_PPO' else\
        sac if alg == 'SAC' else \
        rarl_sac 
    
    list_for_file = []
    for i in range(10):
        rew, attempts, steps, rew_list = Test(RL, env, 10_000)
        list_for_file.extend([{
            'algorithm' : alg,
            'perturbation' : pm_pert,
            'steps' : elem[1],
            'reward' : elem[0],
            'model' : RL.model_name
            } for elem in rew_list])
        
    with open(f'Files/InvertedPendulum/{alg}_{pm_pert}.csv', 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=[k for k in list_for_file[0].keys()])
        writer.writeheader()
        writer.writerows(list_for_file)


if __name__ == '__main__':
    for pert in [-0.9, -0.5, -0.1, 0, 0.1, 0.5, 1, 2]:
        main(render=False, train=True, pm_pert = pert, alg = 'SAC', model_to_load='Models/CartPole_models/Ideal_models/CartPole_model_1')
    #main(render=False, train=True, pm_pert = 0.1, alg = 'SAC') # test SAC