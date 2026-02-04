import torch.nn as nn
import numpy as np
import mujoco
import torch
import torch.nn.functional as F
from torch.distributions import Normal
from torch.distributions import Beta

import PPO_RARL as PPO
import architectures as net
import SAC_RARL
import ENV_Wrapper



    
class HalfCheetah_env_pert(ENV_Wrapper.ENV_Adversarial_wrapper):
    def __init__(self, env_name : str,
                   action_range : list[float, float],
               adv_action_range : list[float, float],
                    render_mode : str    = None,
                is_norm_wrapper : bool   = True,
                algorithm       : str    = 'PPO') -> None:
        super().__init__(env_name, action_range, adv_action_range, render_mode, is_norm_wrapper, algorithm=algorithm)

        self.mj_model, self.mj_data = self.env.unwrapped.model, self.env.unwrapped.data

        self.ids = {}
        self.ids['back_foot'] = self.mj_model.body("bfoot").id
        self.ids['front_foot'] = self.mj_model.body("ffoot").id
        self.ids['floor'] = self.mj_model.geom("floor").id



    def perturbate_env(self, o_act):
        # apply forces on feats
        o_act = self._preprocess_action(o_act)
        self.mj_data.xfrc_applied[self.ids['back_foot']] = np.array([o_act[0], o_act[1], 0.0, 0.0, 0.0, 0.0])
        self.mj_data.xfrc_applied[self.ids['front_foot']] = np.array([o_act[2], o_act[3], 0.0, 0.0, 0.0, 0.0])
        return

    def step(self, action, action_adv = None):
        # make the opponet perturb the environment
        if action_adv != None:
            o_act = self._scale_action(action_adv, self.adv_action_range)
            self.perturbate_env(o_act)

        # perform the action on the environment
        state, reward, terminated, truncated, info = self.env.step(self._preprocess_action(self._scale_action(action, self.action_range)))

        return self._postprocess_state(state), reward, terminated, truncated, info

    def reset(self):
      state, info = self.env.reset()
      return self._postprocess_state(state), info

def Test(RL, env, steps = 10_000):
    
    env.update_running_stats = False 
    s, _ = env.reset()
    reward = 0
    attempts = 1
    for i in range(steps):
        env.update_running_stats = False 
        s, r, term, tronc, _ = env.step(RL.act(s))
        reward += r
        if term or tronc:
            s, _ = env.reset()
            attempts += 1
        if np.mod(i, 1000) == 0: print('#', end='', flush=True)
    env.close()

    print(f'---> rewards: {reward/attempts} | gained in {attempts} attempts')

def Perturbate_env(env, pert = 0):
    if pert == 0: return
    # must be perturbed the walker model

    # modify masses
    for i in [0, 1, 2, 3, 4, 5]:
      print(f"Original {i} mass: {env.unwrapped.model.body_mass[i]}", end='')
      env.unwrapped.model.body_mass[i] += env.unwrapped.model.body_mass[i]*pert
      print(f" ---> New {i} mass: {env.unwrapped.model.body_mass[i]}")

    print(f'original friction: {env.mj_model.geom_friction[env.ids['floor']]}', end='')
    env.mj_model.geom_friction[env.ids['floor']] = [0.01, 0.001, 0.001] # [sliding, torsional, rolling]
    print(f' ---> new friction: {env.mj_model.geom_friction[env.ids['floor']]}')

def main(render = True, train = False, alg = 'PPO', pm_pert = 0):
    # init environment and neural network
    if render:
        render_mode = ENV_Wrapper.ENV_Adversarial_wrapper.HUMAN_RENDER
        
    # init environment and neural networks
    if alg in ['PPO', 'RARL_PPO']:
        
        player = net.HalfCheetah_NN()
        opponent = net.HalfCheetah_NN(n_outputs=4) # 2 output for X, Y forces on both feat
    
    if alg in ['SAC', 'RARL_SAC']:
        
        player = {
            'policy': net.PolicyNetwork_SAC(),
            'Q1_target': net.SoftQNetwork_SAC(),
            'Q2_target': net.SoftQNetwork_SAC(),
            'Q1'    : net.SoftQNetwork_SAC(),
            'Q2'    : net.SoftQNetwork_SAC()
        }
        opponent = {
            'policy': net.PolicyNetwork_SAC(n_outputs=4),
            'Q1_target': net.SoftQNetwork_SAC(n_inputs = 17 + 4),
            'Q2_target': net.SoftQNetwork_SAC(n_inputs = 17 + 4),
            'Q1'    : net.SoftQNetwork_SAC(n_inputs = 17 + 4),
            'Q2'    : net.SoftQNetwork_SAC(n_inputs = 17 + 4)
        }

    if alg in ['PPO', 'SAC'] or (alg in ['RARL_PPO', 'RARL_SAC'] and not train):
        
        env = ENV_Wrapper.ENV_wrapper(
            env_name='HalfCheetah-v5',
            act_min=-1.0,
            act_max=1.0,
            render_mode=render_mode if render else None,
            is_norm_wrapper=True, algorithm= alg)
        
    else:

        env = HalfCheetah_env_pert(
            env_name='HalfCheetah-v5',
            action_range=[-1.0, 1.0],
            adv_action_range=[-0.01, 0.01],
            render_mode=render_mode if render else None,
            is_norm_wrapper=True, algorithm= alg)

    # init the PPO or RARL_PPO algorithm
    if alg == 'RARL_PPO':
        rarl_ppo = PPO.RARL_PPO(player, opponent, env, print_flag=False, lr_player=1e-4, name='Models/HalfCitah_models/Adversarial_models/HalfCheetah_adversarial')
        if train: rarl_ppo.train(player_episode=10,
                             opponent_episode=4,
                             episodes=650,
                             mini_bach=128,
                             max_steps_rollouts=2048,
                             continue_prev_train=True)
        rarl_ppo.load()
    elif alg == 'PPO':
        ppo = PPO.PPO(player, env, print_flag=False, lr=1e-4, name='Models/HalfCitah_models/Ideal_models/HalfCheetah')
        if train: ppo.train(episodes=500, mini_bach=128, max_steps_rollouts=2048, continue_prev_train=True)
        ppo.load()
    
    if alg == 'RARL_SAC':
        rarl_sac = SAC_RARL.RARL_SAC(player, opponent, env, print_flag=False, lr_Q=3e-4, lr_pi=1e-4, name='HalfCheetah_Adversarial_SAC_model')
        if train: rarl_sac.train(player_episode=10, 
                             opponent_episode=4, 
                             episodes=1000, 
                             epoch = 1,
                             mini_batch=128, 
                             max_steps_rollouts=1024, 
                             continue_prev_train=False)
        rarl_sac.load()
        
    elif alg == 'SAC':
        sac = SAC_RARL.SAC(player['Q1_target'], player['Q2_target'], player['Q1'], player ['Q2'], player['policy'], env, print_flag=False, lr_Q=3e-4, lr_pi=1e-4, name='HalfCheetah_model_SAC')
        if train: sac.train(episodes=1000, epoch=1, mini_batch=128, max_steps_rollouts=1024, continue_prev_train=False)
        sac.load()

    env.close()

    # render the simulation if needed
    if not render: return

    # perturbate the modle paramether
    Perturbate_env(env, pm_pert)

    # choise the algorithm for run the simulation
    RL = ppo  if alg == 'PPO' else \
        rarl_ppo if alg == 'RARL_PPO' else\
        sac if alg == 'SAC' else \
        rarl_sac 
        
    Test(RL, env, 10_000)

if __name__ == '__main__':
    #main(render=False, train=True, alg = 'PPO') # train with PPO
    #main(render=False, train=True, alg = 'RARL') # train with RARL
 #   main(render=True, train=False, pm_pert = -0.1, alg = 'PPO') # test PPO
    main(render=False, train=True, pm_pert = -0.1, alg = 'SAC') # test SAC