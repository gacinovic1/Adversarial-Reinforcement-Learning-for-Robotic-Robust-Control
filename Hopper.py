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
    
    
class Hopper_env_pert(ENV_Wrapper.ENV_Adversarial_wrapper):
    def __init__(self, env_name : str,
                   action_range : list[float, float], 
               adv_action_range : list[float, float], 
                    render_mode : str    = None, 
                is_norm_wrapper : bool   = True,
                algorithm : str = 'RARL_SAC') -> None:
        super().__init__(env_name, action_range, adv_action_range, render_mode, is_norm_wrapper, algorithm)

        self.mj_model, self.mj_data = self.env.unwrapped.model, self.env.unwrapped.data
        
        self.ids['foot'] = self.mj_model.body('foot').id
        self.ids['torso']    = self.mj_model.body('torso').id

    def perturbate_env(self, o_act):
        # apply forces on feats
        o_act = self._preprocess_action(o_act)
        force = np.array([o_act[0], o_act[1], 0.0, 0.0, 0.0, 0.0])
        body_id = self.ids['foot']
        self.mj_data.xfrc_applied[self.ids['foot']] = np.array([o_act[0], o_act[1], 0.0, 0.0, 0.0, 0.0])
        self.mj_data.xfrc_applied[self.ids['torso']] = np.array([o_act[2], o_act[3], 0.0, 0.0, 0.0, 0.0])
        self.last_force = force
        self.last_force_body = body_id
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


def Perturbate_env(env, pert = 0, frict = 1.0):
    #if pert == 0: return
    # must be perturbed the walker model

    # modify pendolum mass
    for i in [1]:
      print(f"Original {i} mass: {env.unwrapped.model.body_mass[i]}", end='')
      env.unwrapped.model.body_mass[i] += env.unwrapped.model.body_mass[i]*pert
      print(f" ---> New {i} mass: {env.unwrapped.model.body_mass[i]}")
      new_mass = env.unwrapped.model.body_mass[i].sum()

    model = env.unwrapped.model
    floor_id = mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,"floor")
    print(f'original friction: {env.mj_model.geom_friction[env.ids['floor']]}', end='')
    model.geom_friction[floor_id] = model.geom_friction[floor_id][0] * frict # sliding # [sliding, torsional, rolling]
    print(f' ---> new friction: {env.mj_model.geom_friction[env.ids['floor']]}')
    new_friction = model.geom_friction[floor_id][0]

    return new_mass, new_friction

def main(render = True, train = False, alg = 'RARL', pm_pert = 0, frict = 1.0, model_to_load = '', heatmap = False):
    
    if render:
        render_mode = ENV_Wrapper.ENV_Adversarial_wrapper.HUMAN_RENDER
        
    # init environment and neural networks
    if alg in ['PPO', 'RARL_PPO']:
        
        player = net.Hopper_NN_PPO(n_inputs = 11, n_outputs = 3)
        opponent = net.Hopper_NN_PPO(n_inputs = 11,n_outputs=2) # 2 output for X, Y forces on the foot and torso
    
    if alg in ['SAC', 'RARL_SAC']:
        
        player = {
            'policy': net.PolicyNetwork_SAC(n_inputs= 11, n_outputs=3),
            'Q1_target': net.SoftQNetwork_SAC(n_inputs = 11 + 3),
            'Q2_target': net.SoftQNetwork_SAC(n_inputs = 11 + 3),
            'Q1'    : net.SoftQNetwork_SAC(n_inputs = 11 + 3),
            'Q2'    : net.SoftQNetwork_SAC(n_inputs = 11 + 3)
        }
        opponent = {
            'policy': net.PolicyNetwork_SAC(n_inputs= 11, n_outputs=4),
            'Q1_target': net.SoftQNetwork_SAC(n_inputs = 11 + 4),
            'Q2_target': net.SoftQNetwork_SAC(n_inputs = 11 + 4),
            'Q1'    : net.SoftQNetwork_SAC(n_inputs = 11 + 4),
            'Q2'    : net.SoftQNetwork_SAC(n_inputs = 11 + 4)
        }

    if alg in ['PPO', 'SAC'] or (alg in ['RARL_PPO', 'RARL_SAC'] and not train):
        
        env = ENV_Wrapper.ENV_wrapper(
            env_name='Hopper-v5',
            act_min=-1.0,
            act_max=1.0,
            render_mode='' if render else None,
            is_norm_wrapper=False, algorithm= alg)
        
    else:

        env = Hopper_env_pert(
            env_name='Hopper-v5',
            action_range=[-1.0, 1.0],
            adv_action_range=[-0.01, 0.01],
            render_mode=render_mode if render else None,
            is_norm_wrapper=True, algorithm= alg)

        

    # init the PPO or RARL_PPO algorithm
    if alg == 'RARL_PPO':
        rarl_ppo = PPO.RARL_PPO(player, opponent, env, print_flag=False, lr_player=1e-4, name='Models/Hopper/Adversarial_models/Hopper_Adversarial_PPO_model')
        if train: rarl_ppo.train(player_episode=10, 
                             opponent_episode=4, 
                             episodes=10, 
                             mini_bach=128, 
                             max_steps_rollouts=2048, 
                             continue_prev_train=False)
        rarl_ppo.load()
        
    elif alg == 'PPO':
        ppo = PPO.PPO(player, env, print_flag=False, lr=1e-4, name='Models/Hopper/Ideal_models/Hopper_model_PPO')
        if train: ppo.train(episodes=10, mini_bach=128, max_steps_rollouts=2048, continue_prev_train=False)
        ppo.load()

    if alg == 'RARL_SAC':
        rarl_sac = SAC_RARL.RARL_SAC(player, opponent, env, print_flag=False, lr_Q=3e-4, lr_pi=1e-4, name='Hopper_Adversarial_SAC_model')
        if train: rarl_sac.train(player_episode=10, 
                             opponent_episode=4, 
                             episodes=400, 
                             epoch = 1,
                             mini_batch=128, 
                             max_steps_rollouts=1024, 
                             continue_prev_train=False)
        rarl_sac.load()
        
    elif alg == 'SAC':
        sac = SAC_RARL.SAC(player['Q1_target'], player['Q2_target'], player['Q1'], player ['Q2'], player['policy'], env, print_flag=False, lr_Q=3e-4, lr_pi=1e-4, name='Hopper_model_SAC')
        if train: sac.train(episodes=1000, epoch=1, mini_batch=128, max_steps_rollouts=1024, continue_prev_train=False)
        sac.load()

    env.close()

    # render the simulation if needed
    if not render: return
    
    # choise the algorithm for run the simulation
    RL = ppo  if alg == 'PPO' else \
        rarl_ppo if alg == 'RARL_PPO' else\
        sac if alg == 'SAC' else \
        rarl_sac 
            
    # perturbate the modle paramether
    new_mass, new_friction = Perturbate_env(env, pm_pert, frict)

    # choise the algorithm for run the simulation
    RL = ppo if alg == 'PPO' else rarl_ppo
    
    list_for_file = []
    for i in range(3):
        rew, attempts, steps, rew_list = Test(RL, env, 2_000)
        list_for_file.extend([{
            'algorithm' : alg,
            'Mass' : new_mass,
            'Friction': new_friction,
            'steps' : elem[1],
            'reward' : elem[0],
            'model' : RL.model_name
            } for elem in rew_list])
    token = model_to_load.split("/")[-1]

    perturbation = '' + ('Mass_' if pm_pert != 0.0 else '') + ('Friction_' if frict != 1.0 else '')

    if not heatmap:
        file = f'Files/Hopper_2/{alg}_{perturbation}{new_mass:0.4f}_{new_friction:0.4f}_{token}.csv'
    else:
        file = f'Files/Hopper_2/heatmap/{alg}_{perturbation}{new_mass:0.4f}_{new_friction:0.4f}_{token}_heatmap.csv'
    
    with open(file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=[k for k in list_for_file[0].keys()])
        writer.writeheader()
        writer.writerows(list_for_file)

if __name__ == '__main__':
    #main(render=False, train=True, alg = 'PPO') # train with PPO
    #main(render=False, train=True, alg = 'RARL') # train with RARL
    #main(render=True, train=False, pm_pert = 0, alg = 'PPO') # test PPO
    #main(render=True, train=False, pm_pert = 0, alg = 'RARL_PPO') # test RARL PPO
    #main(render=False, train=True, pm_pert = 1, alg = 'RARL_SAC') # test SAC
  #  main(render=False, train=True, pm_pert = 1, alg = 'RARL_SAC') # test RARL SAC

    for path, alg in zip(['Idela-models/Hopper_model_PPO', 'Adversarial_models/Hopper_Adversarial_PPO_model'], ['PPO', 'RARL_PPO']):
            for pert in [-0.9, -0.7, -0.5, -0.3, -0.1, 0.0 ,0.2, 0.5, 0.7, 0.9, 1]: #  
                for frict in [0.0, 0.1, 0.4, 0.8, 1.0, 1.3, 1.7, 2.0, 2.2, 2.5]:
                    main(render=True, train=False, pm_pert = pert, frict=frict, alg = alg, model_to_load = f'Models/Hopper/' + path, heatmap = True) # test RARL_PPO
    
