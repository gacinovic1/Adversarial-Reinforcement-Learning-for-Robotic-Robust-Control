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
    

class ValueNetwork_SAC(nn.Module):
    def __init__(self, n_inputs = 17, n_outputs = 1, init_w=3e-3) -> None:
        super().__init__()

        self.linear1 = nn.Linear(n_inputs , 256)
        self.linear2 = nn.Linear(256, 256)
        self.linear3 = nn.Linear(256, 1)

        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class SoftQNetwork_SAC(nn.Module):
    def __init__(self, n_inputs = 17 + 6, n_outputs = 1, init_w=3e-3) -> None:
        super().__init__()

        self.linear1 = nn.Linear(n_inputs, 256)
        self.linear2 = nn.Linear(256, 256)
        self.linear3 = nn.Linear(256, 1)

        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class PolicyNetwork_SAC(nn.Module):
    def __init__(self, n_inputs = 17, n_outputs = 6, init_w=3e-4) -> None:
        super().__init__()

        self.linear1 = nn.Linear(n_inputs, 256)
        self.linear2 = nn.Linear(256, 256)

        self.mean_linear = nn.Linear(256, n_outputs)
        self.mean_linear.weight.data.uniform_(-init_w, init_w)
        self.mean_linear.bias.data.uniform_(-init_w, init_w)

        self.log_std_linear = nn.Linear(256, n_outputs) 
        self.log_std_linear.weight.data.uniform_(-init_w, init_w)
        self.log_std_linear.bias.data.uniform_(-init_w, init_w)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, -5, 1)

        return mean, log_std
    

class Walker_env_pert(ENV_Wrapper.ENV_Adversarial_wrapper):
    def __init__(self, env_name : str,
                   action_range : list[float, float], 
               adv_action_range : list[float, float], 
                    render_mode : str    = None, 
                is_norm_wrapper : bool   = True,
                algorithm : str = 'SAC') -> None:
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

def Test(RL, env, steps = 10_000):
    s, _ = env.reset()
    reward = 0
    attempts = 1
    for i in range(steps):
        s, r, term, tronc, _ = env.step(RL.act(s))
        reward += r
        if term or tronc: 
            s, _ = env.reset()
            attempts += 1
        if np.mod(i, 1000) == 0: print('#', end='', flush=True)
    env.close()

    print(f'\n---> rewards: {reward/attempts} | gained in {attempts} attempts')


def Perturbate_env(env, pert = 0):
    # must be perturbed the walker model
    print(f"Original mass: {env.unwrapped.model.body_mass[2]}")

    # modify body's masses
    env.unwrapped.model.body_mass[2] += env.unwrapped.model.body_mass[2]*pert
    print(f"New mass: {env.unwrapped.model.body_mass[2]}")

    env.mj_model.geom_friction[env.ids['floor']] = [0.01, 0.01, 0.01] # [sliding, torsional, rolling]


def main(render = True, train = False, alg = 'RARL', pm_pert = 0):
    
    if render:
        render_mode = ENV_Wrapper.ENV_Adversarial_wrapper.HUMAN_RENDER
        
    # init environment and neural networks
    if alg in ['PPO', 'RARL_PPO']:
        
        player = Walker_NN_PPO()
        opponent = Walker_NN_PPO(n_outputs=4) # 2 output for X, Y forces on both feat
    
    if alg in ['SAC', 'RARL_SAC']:
        
        player = {
            'policy': PolicyNetwork_SAC(),
            'value' : ValueNetwork_SAC(),
            'value_target': ValueNetwork_SAC(),
            'Q1'    : SoftQNetwork_SAC(),
            'Q2'    : SoftQNetwork_SAC()
        }
        opponent = {
            'policy': PolicyNetwork_SAC(),
            'value' : ValueNetwork_SAC(),
            'value_target': ValueNetwork_SAC(),
            'Q1'    : SoftQNetwork_SAC(),
            'Q2'    : SoftQNetwork_SAC()
        }

    if alg in ['PPO', 'SAC']:
        
        env = ENV_Wrapper.ENV_wrapper(
        env_name='Walker2d-v5',
        act_min=-1.0,
        act_max=1.0,
        render_mode=render_mode if render else None,
        is_norm_wrapper=True, algorithm= alg)
        
    else:
        
        env = Walker_env_pert(
        env_name='Walker2d-v5',
        action_range=[-1.0, 1.0],
        adv_action_range=[-0.01, 0.001],
        render_mode=render_mode if render else None,
        is_norm_wrapper=True, algorithm= alg)
        

    # init the PPO or RARL_PPO algorithm
    if alg == 'RARL_PPO':
        rarl_ppo = PPO.RARL_PPO(player, opponent, env, print_flag=False, lr_player=1e-4, name='Walker_2D_Adversarial_PPO_model')
        if train: rarl_ppo.train(player_episode=10, 
                             opponent_episode=4, 
                             episodes=10, 
                             mini_bach=128, 
                             max_steps_rollouts=2048, 
                             continue_prev_train=False)
        rarl_ppo.load()
        
    elif alg == 'PPO':
        ppo = PPO.PPO(player, env, print_flag=False, lr=1e-4, name='Walker_2D_model_PPO')
        if train: ppo.train(episodes=10, mini_bach=128, max_steps_rollouts=2048, continue_prev_train=False)
        ppo.load()

    if alg == 'RARL_SAC':
        rarl_sac = SAC_RARL.RARL_SAC(player, opponent, env, print_flag=False, lr_player=1e-4, name='Walker_2D_Adversarial_SAC_model')
        if train: rarl_sac.train(player_episode=10, 
                             opponent_episode=4, 
                             episodes=10, 
                             mini_bach=128, 
                             max_steps_rollouts=2048, 
                             continue_prev_train=False)
        rarl_sac.load()
        
    elif alg == 'SAC':
        sac = SAC_RARL.SAC(player['value'], player['value_target'], player['Q1'], player ['Q2'], player['policy'], env, print_flag=False, lr_V=1e-4, lr_Q=1e-4, lr_pi=1e-4, name='Walker_2D_model_SAC')
        if train: sac.train(episodes=1000, epoch=1, mini_batch=128, max_steps_rollouts=1024, continue_prev_train=False)
        sac.load()

    env.close()

    # render the simulation if needed
    if not render: return

    # perturbate the modle paramether
    Perturbate_env(env, pm_pert)
    
    # choise the algorithm for run the simulation
    RL = ppo if alg == 'PPO' else rarl_ppo
    Test(RL, env, 10_000)

if __name__ == '__main__':
    #main(render=False, train=True, alg = 'PPO') # train with PPO
    #main(render=False, train=True, alg = 'RARL') # train with RARL
 #   main(render=False, train=True, pm_pert = 1, alg = 'PPO') # test PPO
 #   main(render=False, train=True, pm_pert = 1, alg = 'RARL_PPO') # test RARL PPO
    main(render=False, train=True, pm_pert = 1, alg = 'SAC') # test SAC
  #  main(render=False, train=True, pm_pert = 1, alg = 'RARL_SAC') # test RARL SAC
    
