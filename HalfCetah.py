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

import csv

class HalfCetah_NN(nn.Module):
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
    
class HalfCetah_env_pert(ENV_Wrapper.ENV_Adversarial_wrapper):
    def __init__(self, env_name : str,
                   action_range : list[float, float],
               adv_action_range : list[float, float],
                    render_mode : str    = None,
                is_norm_wrapper : bool   = True,
                algorithm       : str    = 'PPO') -> None:
        super().__init__(env_name, action_range, adv_action_range, render_mode, is_norm_wrapper, algorithm=algorithm)

        self.mj_model, self.mj_data = self.env.unwrapped.model, self.env.unwrapped.data

        self.ids = {}
        self.ids['bacck_foot'] = self.mj_model.body("bfoot").id
        self.ids['fron_foot' ] = self.mj_model.body("ffoot").id
        self.ids['floor'] = self.mj_model.geom("floor").id



    def perturbate_env(self, o_act):
        # apply forces on feats
        o_act = self._preprocess_action(o_act)
        self.mj_data.xfrc_applied[self.ids['bacck_foot']] = np.array([o_act[0], o_act[1], 0.0, 0.0, 0.0, 0.0])
        self.mj_data.xfrc_applied[self.ids['fron_foot' ]] = np.array([o_act[2], o_act[3], 0.0, 0.0, 0.0, 0.0])
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
    for i in [3]:
      print(f"Original {i} mass: {env.unwrapped.model.body_mass[i]}", end='')
      env.unwrapped.model.body_mass[i] += env.unwrapped.model.body_mass[i]*pert
      print(f" ---> New {i} mass: {env.unwrapped.model.body_mass[i]}")
      new_mass = env.unwrapped.model.body_mass[i].sum()

    model = env.unwrapped.model
    floor_id = mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,"floor")
    print(f'original friction: {env.mj_model.geom_friction[env.ids['floor']]}', end='')
    for i in range(1):
        model.geom_friction[floor_id][i] = model.geom_friction[floor_id][i] * frict # sliding # [sliding, torsional, rolling]
    print(f' ---> new friction: {env.mj_model.geom_friction[env.ids['floor']]}')
    new_friction = sum(model.geom_friction[floor_id])

    return new_mass, new_friction

def main(render = True, train = False, alg = 'RARL', pm_pert = 0, frict = 1.0, model_to_load = '', heatmap = False):
    # init environment and neural network
    env = HalfCetah_env_pert(
        env_name='HalfCheetah-v5',
        action_range=[-1, 1],
        adv_action_range=[-0.01, 0.01],
        render_mode='',
        is_norm_wrapper=False
        )

    player = HalfCetah_NN()
    opponent = HalfCetah_NN(n_outputs=4) # 2 output for X, Y forces on both feat

    # init the PPO or RARL_PPO algorithm
    if alg == 'RARL':
        rarl_ppo = PPO.RARL_PPO(player, opponent, env, print_flag=False, lr_player=1e-3, name='Models/HalfCitah_models/Adversarial_models/HalfCheetah_adversarial_2')
        if train: rarl_ppo.train(player_episode=10,
                             opponent_episode=4,
                             episodes=700,
                             mini_bach=128,
                             max_steps_rollouts=2048,
                             continue_prev_train=False)
        rarl_ppo.load()
    elif alg == 'PPO':
        ppo = PPO.PPO(player, env, print_flag=False, lr=1e-4, name='Models/HalfCitah_models/Ideal_models/HalfCheetah')
        if train: ppo.train(episodes=500, mini_bach=128, max_steps_rollouts=2048, continue_prev_train=True)
        ppo.load()

    env.close()

    # render the simulation if needed
    if not render: return

    # perturbate the modle paramether
    new_mass, new_friction = Perturbate_env(env, pm_pert, frict)

    # choise the algorithm for run the simulation
    RL = ppo if alg == 'PPO' else rarl_ppo
    
    list_for_file = []
    for i in range(5):
        rew, attempts, steps, rew_list = Test(RL, env, 3_000)
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
        file = f'Files/Half_C/{alg}_{perturbation}{new_mass:0.4f}_{new_friction:0.4f}_{token}.csv'
    else:
        file = f'Files/Half_C/heatmap/{alg}_{perturbation}{new_mass:0.4f}_{new_friction:0.4f}_{token}_heatmap.csv'
    
    with open(file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=[k for k in list_for_file[0].keys()])
        writer.writeheader()
        writer.writerows(list_for_file)

if __name__ == '__main__':
    #main(render=False, train=True, alg = 'PPO') # train with PPO
    #main(render=False, train=True, alg = 'RARL') # train with RARL
    #main(render=True, train=False, pm_pert = -0.1, alg = 'PPO') # test PPO
    #main(render=True, train=False, pm_pert = -0.1, alg = 'RARL') # test RARL

    for path, alg in zip(['Idela-models/HalfCheetah', 'Adversarial_models/HalfCeetah_adversarial_2'], ['PPO', 'RARL']):
        for pert in [-0.9, -0.7, -0.5, -0.3, -0.1, 0.0 ,0.2, 0.5, 0.7, 0.9, 1]: #  
            for frict in [0.0, 0.1, 0.4, 0.8, 1.0, 1.3, 1.7, 2.0, 2.2, 2.5]:
                main(render=True, train=False, pm_pert = pert, frict=frict, alg = alg, model_to_load = f'Models/HalfCitah_models/' + path, heatmap = True) # test RARL_PPO
                