from pyexpat import model
import torch.nn as nn
import numpy as np
import mujoco
import torch
import torch.nn.functional as F
from torch.distributions import Normal
from torch.distributions import Beta

import PPO_RARL as PPO
import SAC_RARL
import ENV_Wrapper
import architectures as net
import csv
    
class Walker_env_pert(ENV_Wrapper.ENV_Adversarial_wrapper):
    def __init__(self, env_name : str,
                   action_range : list[float, float], 
               adv_action_range : list[float, float], 
                    render_mode : str    = None, 
                is_norm_wrapper : bool   = True,
                algorithm : str = 'RARL_SAC') -> None:
        super().__init__(env_name, action_range, adv_action_range, render_mode, is_norm_wrapper, algorithm)

        self.mj_model, self.mj_data = self.env.unwrapped.model, self.env.unwrapped.data
        
        self.ids = {}
        self.ids['torso'     ] = self.mj_model.body("torso").id
        self.ids['left_foot' ] = self.mj_model.body("foot_left").id
        self.ids['right_foot'] = self.mj_model.body("foot").id
        self.ids['floor'     ] = self.mj_model.geom("floor").id

    def perturbate_env(self, o_act):
        # apply forces on feats
        o_act = self._preprocess_action(o_act)
        self.mj_data.xfrc_applied[self.ids['left_foot' ]] = np.array([o_act[0], o_act[1], 0.0, 0.0, 0.0, 0.0])
        self.mj_data.xfrc_applied[self.ids['right_foot']] = np.array([o_act[2], o_act[3], 0.0, 0.0, 0.0, 0.0])
        return

    def step(self, action, action_adv = None):
        # make the opponet perturb the environment
        if action_adv != None:
            o_act = self._scale_action(action_adv, self.adv_action_range)
            self.perturbate_env(o_act)

        # perform the action on the environment
        state, reward, terminated, truncated, info = self.env.step(self._preprocess_action(self._scale_action(action, self.action_range)))

        return self._postprocess_state(state), reward, terminated, truncated, info

def Test(RL, env, steps = 10_000) -> tuple[float, int, int, list[float, int]]:
    s, _ = env.reset()
    rew_list = [(0, 0)]
    reward = 0
    attempts = 0
    for i in range(steps):
        env.update_running_stats = False 
        s, r, term, tronc, _ = env.step(RL.act(s))
        reward += r
        if term or tronc: 
            s, _ = env.reset()
            attempts += 1
            rew_list.append((reward - np.sum([r[0] for r in rew_list]), i+1 - np.sum([s[1] for s in rew_list])))
        if np.mod(i, 1000) == 0: print('#', end='', flush=True)
    env.close()

    print(f'\n---> rewards: {reward/attempts} | gained in {attempts} attempts')

    return (reward, attempts, steps, rew_list[1:])

    print(f'\n---> rewards: {reward/attempts} | gained in {attempts} attempts')


def Perturbate_env(env, pert = 0):
    if pert == 0: return
    # must be perturbed the walker model

    # modify pendolum mass
    for i in [0, 1, 2, 3, 4, 5]:
      print(f"Original {i} mass: {env.unwrapped.model.body_mass[i]}", end='')
      env.unwrapped.model.body_mass[i] += env.unwrapped.model.body_mass[i]*pert
      print(f" ---> New {i} mass: {env.unwrapped.model.body_mass[i]}")

    model = env.unwrapped.model

    floor_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_GEOM,
    "floor"
    )

    model.geom_friction[floor_id] = np.array([0.01, 0.01, 0.01])# [sliding, torsional, rolling]


def main(render = True, train = False, alg = 'RARL', pm_pert = 0, model_to_load = ''):
    
    if render:
        render_mode = ''#ENV_Wrapper.ENV_Adversarial_wrapper.HUMAN_RENDER
        
    # init environment and neural networks
    if alg in ['PPO', 'RARL_PPO', 'RARL']:
        
        player = net.Walker_NN_PPO()
        opponent = net.Walker_NN_PPO(n_outputs=4) # 2 output for X, Y forces on both feat
    
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

    if alg in ['SAC']:
        
        env = ENV_Wrapper.ENV_wrapper(
            env_name='Walker2d-v5',
            act_min=-1.0,
            act_max=1.0,
            render_mode=render_mode if render else None,
            is_norm_wrapper=False, algorithm= alg)
        
    else:

        env = Walker_env_pert(
            env_name='Walker2d-v5',
            action_range=[-1.0, 1.0],
            adv_action_range=[-0.01, 0.001],
            render_mode=render_mode if render else None,
            is_norm_wrapper=False, algorithm= alg)
        

    # init the PPO or RARL_PPO algorithm
    if alg in ['RARL_PPO', 'RARL']:
        rarl_ppo = PPO.RARL_PPO(player, opponent, env, print_flag=False, lr_player=1e-4, name='Walker_2D_Adversarial_PPO_model' if model_to_load == '' else model_to_load)
        if train: rarl_ppo.train(player_episode=10, 
                             opponent_episode=4, 
                             episodes=10, 
                             mini_bach=128, 
                             max_steps_rollouts=2048, 
                             continue_prev_train=False)
        rarl_ppo.load()
        
    elif alg == 'PPO':
        ppo = PPO.PPO(player, env, print_flag=False, lr=1e-4, name='Walker_2D_model_PPO' if model_to_load == '' else model_to_load)
        if train: ppo.train(episodes=10, mini_bach=128, max_steps_rollouts=2048, continue_prev_train=False)
        ppo.load()

    if alg == 'RARL_SAC':
        rarl_sac = SAC_RARL.RARL_SAC(player, opponent, env, print_flag=False, lr_Q=3e-4, lr_pi=1e-4, name='Walker_2D_Adversarial_SAC_model')
        if train: rarl_sac.train(player_episode=10, 
                             opponent_episode=4, 
                             episodes=1000, 
                             epoch = 1,
                             mini_batch=128, 
                             max_steps_rollouts=1024, 
                             continue_prev_train=False)
        rarl_sac.load()
        
    elif alg == 'SAC':
        sac = SAC_RARL.SAC(player['Q1_target'], player['Q2_target'], player['Q1'], player ['Q2'], player['policy'], env, print_flag=False, lr_Q=3e-4, lr_pi=1e-4, name='Walker_2D_model_SAC')
        if train: sac.train(episodes=1000, epoch=1, mini_batch=128, max_steps_rollouts=1024, continue_prev_train=False)
        sac.load()

    env.close()

    # render the simulation if needed
    if not render: return

    # perturbate the model paramether
    Perturbate_env(env, pm_pert)
    
    # choise the algorithm for run the simulation
    RL = ppo if alg == 'PPO' else rarl_ppo
    #RL = sac if alg == 'SAC' else rarl_sac
    
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
    token = model_to_load.split("/")[-1]
    with open(f'Files/Walker2D/{alg}_{pm_pert}_{token}.csv', 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=[k for k in list_for_file[0].keys()])
        writer.writeheader()
        writer.writerows(list_for_file)

if __name__ == '__main__':
    #main(render=False, train=True, alg = 'PPO') # train with PPO
    #main(render=False, train=True, alg = 'RARL') # train with RARL
    #main(render=True, train=False, pm_pert = 0, alg = 'PPO', model_to_load = 'Models/CartPole_models/Ideal_models/CartPole_model_1') # test PPO
 #   main(render=False, train=True, pm_pert = 1, alg = 'RARL_PPO') # test RARL PPO
    #main(render=False, train=True, pm_pert = 1, alg = 'SAC') # test SAC
  #  main(render=False, train=True, pm_pert = 1, alg = 'RARL_SAC') # test RARL SAC
    
    for file in ["Walker_feet_model", "Walker_feet_model_01", "Walker_feet_model_05"]:
        for pert in [-0.9, -0.5, -0.1, 0, 0.1, 0.5, 1, 2]:
            main(render=True, train=False, pm_pert = pert, alg = 'RARL', model_to_load = f'Models/Walker_models/Adversarial_models/{file}') # test PPO
