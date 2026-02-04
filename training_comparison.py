import numpy as np
import matplotlib.pyplot as plt
import torch
import PPO_RARL as PPO
import SAC_RARL
import ENV_Wrapper
import architectures as net
import csv
import os

from Walker_2D_Adv  import Walker2d_env_pert
from HalfCheetah import HalfCheetah_env_pert
from Hopper import Hopper_env_pert
from Swimmer import Swimmer_env_pert
from Inverted_pendulum import CartPole

def load_rewards_matrix_csv(filename):
    with open(filename, mode='r') as f:
        reader = csv.reader(f)
        header = next(reader)

        data = []
        for row in reader:
            data.append([float(x) for x in row[1:]])  # skip seed

    return np.array(data)  # shape: (n_seeds, episodes)


def save_rewards_matrix_csv(rewards_matrix, filename):
    """
    rewards_matrix: np.array shape (n_seeds, episodes)
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    n_seeds, episodes = rewards_matrix.shape

    with open(filename, mode='w', newline='') as f:
        writer = csv.writer(f)

        header = ['seed'] + [f'ep_{i}' for i in range(episodes)]
        writer.writerow(header)

        for seed in range(n_seeds):
            writer.writerow([seed] + rewards_matrix[seed].tolist())
            
def load_rewards_matrix_csv(filename):
    with open(filename, mode='r') as f:
        reader = csv.reader(f)
        header = next(reader)

        data = []
        for row in reader:
            data.append([float(x) for x in row[1:]])  # skip seed

    return np.array(data)  # shape: (n_seeds, episodes)


def run_comparison(alg_base: str,alg_rarl: str,environment: str,n_seeds: int = 10,episodes: int = 500):
    
    if alg_base not in ['PPO', 'SAC']: 
        raise ValueError("alg_base must be either 'PPO' or 'SAC'")
    
    if alg_rarl not in ['RARL_PPO', 'RARL_SAC']:
        raise ValueError("alg_rarl must be either 'RARL_PPO' or 'RARL_SAC'")

    pert_env_class = globals()[f'{environment}_env_pert']
    
    ENV_PERT_MAP = {
    'Walker2d': Walker2d_env_pert,
    'HalfCheetah': HalfCheetah_env_pert,
    'Hopper': Hopper_env_pert,
    'Swimmer': Swimmer_env_pert,
    'CartPole': CartPole
   }
    
    pert_env_class = ENV_PERT_MAP[environment]

    rewards_base = np.zeros((n_seeds, episodes))
    rewards_rarl = np.zeros((n_seeds, episodes))    

    for seed in range(n_seeds):
        print(f'\n=== Seed {seed} ===')

        np.random.seed(seed)
        torch.manual_seed(seed)

        if alg_base == 'PPO':
            player = net.Walker_NN_PPO()

            env = ENV_Wrapper.ENV_wrapper(
                env_name=f'{environment}-v5',
                act_min=-1.0,
                act_max=1.0,
                render_mode=None,
                is_norm_wrapper=True,
                algorithm=alg_base
            )

            env.unwrapped.reset(seed=seed)

            algo = PPO.PPO(
                player,
                env,
                print_flag=False,
                lr=1e-4,
                name=f'{environment}_{alg_base}_seed_{seed}'
            )

            rewards = algo.train(
                episodes=episodes,
                mini_bach=128,
                max_steps_rollouts=1000,
                continue_prev_train=False
            )

        elif alg_base == 'SAC':
            player = {
                'policy': net.PolicyNetwork_SAC(),
                'Q1_target': net.SoftQNetwork_SAC(),
                'Q2_target': net.SoftQNetwork_SAC(),
                'Q1': net.SoftQNetwork_SAC(),
                'Q2': net.SoftQNetwork_SAC()
            }

            env = ENV_Wrapper.ENV_wrapper(
                env_name=f'{environment}-v5',
                act_min=-1.0,
                act_max=1.0,
                render_mode=None,
                is_norm_wrapper=True,
                algorithm=alg_base
            )

            env.unwrapped.reset(seed=seed)

            algo = SAC_RARL.SAC(
                player['Q1_target'], player['Q2_target'],
                player['Q1'], player['Q2'],
                player['policy'],
                env,
                print_flag=False,
                lr_Q=3e-4,
                lr_pi=1e-4,
                name=f'{environment}_{alg_base}_seed_{seed}'
            )

            rewards = algo.train(
                episodes=episodes,
                epoch=1,
                mini_batch=128,
                max_steps_rollouts=1000,
                continue_prev_train=False
            )
            
        
        rewards = np.array(rewards)

        # sicurezza: tronco/padding se serve
        if len(rewards) < episodes:
            rewards = np.pad(rewards, (0, episodes - len(rewards)), mode='edge')
        else:
            rewards = rewards[:episodes]

        rewards_base[seed] = rewards


        # ================= RARL =================
        if alg_rarl == 'RARL_PPO':
            player = net.Walker_NN_PPO()
            opponent = net.Walker_NN_PPO(n_outputs=4)

            env = pert_env_class(
                env_name=f'{environment}-v5',
                action_range=[-1.0, 1.0],
                adv_action_range=[-0.01, 0.001],
                render_mode=None,
                is_norm_wrapper=True,
                algorithm=alg_rarl
            )

            env.env.unwrapped.reset(seed=seed)

            algo = PPO.RARL_PPO(
                player,
                opponent,
                env,
                print_flag=False,
                lr_player=1e-4,
                name=f'{environment}_{alg_rarl}_seed_{seed}'
            )

            rewards = algo.train(
                player_episode=10,
                opponent_episode=4,
                episodes=episodes,
                mini_bach=128,
                max_steps_rollouts=1000,
                continue_prev_train=False
            )

        elif alg_rarl == 'RARL_SAC':
            player = {
                'policy': net.PolicyNetwork_SAC(),
                'Q1_target': net.SoftQNetwork_SAC(),
                'Q2_target': net.SoftQNetwork_SAC(),
                'Q1': net.SoftQNetwork_SAC(),
                'Q2': net.SoftQNetwork_SAC()
            }

            opponent = {
                'policy': net.PolicyNetwork_SAC(n_outputs=4),
                'Q1_target': net.SoftQNetwork_SAC(n_inputs=21),
                'Q2_target': net.SoftQNetwork_SAC(n_inputs=21),
                'Q1': net.SoftQNetwork_SAC(n_inputs=21),
                'Q2': net.SoftQNetwork_SAC(n_inputs=21)
            }

            env = pert_env_class(
                env_name=f'{environment}-v5',
                action_range=[-1.0, 1.0],
                adv_action_range=[-0.01, 0.001],
                render_mode=None,
                is_norm_wrapper=True,
                algorithm=alg_rarl
            )

            env.env.unwrapped.reset(seed=seed)

            algo = SAC_RARL.RARL_SAC(
                player,
                opponent,
                env,
                print_flag=False,
                lr_Q=3e-4,
                lr_pi=1e-4,
                name=f'{environment}_{alg_rarl}_seed_{seed}'
            )

            rewards = algo.train(
                player_episode=10,
                opponent_episode=4,
                episodes=episodes,
                epoch=1,
                mini_batch=128,
                max_steps_rollouts=1000,
                continue_prev_train=False
            )

            rewards = np.array(rewards)

        if len(rewards) < episodes:
            rewards = np.pad(rewards, (0, episodes - len(rewards)), mode='edge')
        else:
            rewards = rewards[:episodes]

        rewards_rarl[seed] = rewards
        
    base_csv = f"results/{environment}/{alg_base}_rewards.csv"
    rarl_csv = f"results/{environment}/{alg_rarl}_rewards.csv"

    save_rewards_matrix_csv(rewards_base, base_csv)
    save_rewards_matrix_csv(rewards_rarl, rarl_csv)
    
    base_mean = rewards_base.mean(axis=0)
    base_std  = rewards_base.std(axis=0)

    rarl_mean = rewards_rarl.mean(axis=0)
    rarl_std  = rewards_rarl.std(axis=0)

    x = np.arange(episodes)
    
    plt.figure(figsize=(7, 5))

    plt.plot(x, base_mean, color='green', label=alg_base)
    plt.fill_between(x, base_mean - base_std, base_mean + base_std,
                     color='green', alpha=0.3)

    plt.plot(x, rarl_mean, color='blue', label=alg_rarl)
    plt.fill_between(x, rarl_mean - rarl_std, rarl_mean + rarl_std,
                     color='blue', alpha=0.3)

    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.title(environment)
    plt.legend(loc="upper left")
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(f'{alg_base}_vs_RARL_{alg_base}_training_comparison.png', dpi=300)
    plt.close()
    

def main():
    
    run_comparison(alg_base='PPO',alg_rarl='RARL_PPO',environment='Walker2d', n_seeds=5,episodes=500)    
    
if __name__ == '__main__':

    main()