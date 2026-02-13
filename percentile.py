import numpy as np
import matplotlib.pyplot as plt
import torch
import PPO_RARL as PPO
import SAC_RARL
import ENV_Wrapper
import architectures as net
import csv
import os
import random

import Walker_2D_Adv as Walker
from Walker_2D_Adv  import Walker2d_env_pert
from HalfCheetah import HalfCheetah_env_pert
from Hopper import Hopper_env_pert
from Swimmer import Swimmer_env_pert
from Inverted_pendulum import CartPole


def percentile_curve(rewards):
    
    rewards = np.sort(rewards)
    percentiles = np.linspace(0, 100, len(rewards))
    return percentiles, rewards

def run_percentile_test(environment: str,alg: str,n_tests: int = 10):

    assert alg in ['PPO', 'SAC']

    rarl_alg = f'RARL_{alg}'

    rewards_base = []
    rewards_rarl = []

    for seed in range(n_tests):

        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

        env = ENV_Wrapper.ENV_wrapper(
            env_name=f'{environment}-v5',
            act_min=-1.0,
            act_max=1.0,
            render_mode=None,
            is_norm_wrapper=True,
            algorithm=alg
        )
        Walker.Perturbate_env(env, 0.7, 1.4)

        if alg == 'PPO':
            player = net.Walker_NN_PPO()
            agent = PPO.PPO(
                player,
                env,
                print_flag=False,
                name=f'{environment}_{alg}_model'
            )
        else:  # SAC
            player = {
                'policy': net.PolicyNetwork_SAC(),
                'Q1_target': net.SoftQNetwork_SAC(),
                'Q2_target': net.SoftQNetwork_SAC(),
                'Q1': net.SoftQNetwork_SAC(),
                'Q2': net.SoftQNetwork_SAC()
            }
            agent = SAC_RARL.SAC(
                player['Q1_target'], player['Q2_target'],
                player['Q1'], player['Q2'],
                player['policy'],
                env,
                print_flag=False,
                name=f'{environment}_{alg}_model'
            )

        agent.load(model_name = f"{environment}_{alg}_seed_0.pt")
        reward = Walker.Test(agent, env, steps = 1_000)[0]
        rewards_base.append(reward)

        if alg == 'PPO':
            player = net.Walker_NN_PPO()
            opponent = net.Walker_NN_PPO(n_outputs=4)

            agent_rarl = PPO.RARL_PPO(
                player,
                opponent,
                env,
                print_flag=False,
                name=f'{environment}_Adversarial_{alg}_model'
            )

        else:  # RARL_SAC
            player = {
                'policy': net.PolicyNetwork_SAC(),
                'Q1_target': net.SoftQNetwork_SAC(),
                'Q2_target': net.SoftQNetwork_SAC(),
                'Q1': net.SoftQNetwork_SAC(),
                'Q2': net.SoftQNetwork_SAC()
            }
            opponent = {
                'policy': net.PolicyNetwork_SAC(n_outputs=4),
                'Q1_target': net.SoftQNetwork_SAC(n_inputs=17 + 4),
                'Q2_target': net.SoftQNetwork_SAC(n_inputs=17 + 4),
                'Q1': net.SoftQNetwork_SAC(n_inputs=17 + 4),
                'Q2': net.SoftQNetwork_SAC(n_inputs=17 + 4)
            }

            agent_rarl = SAC_RARL.RARL_SAC(
                player,
                opponent,
                env,
                print_flag=False,
                name=f'{environment}_Adversarial_{alg}_model'
            )

        agent_rarl.load(model_name = f"{environment}_RARL_{alg}_seed_0.pt")
        reward = Walker.Test(agent_rarl, env, steps = 1_000)[0]
        rewards_rarl.append(reward)

    p_b, r_b = percentile_curve(rewards_base)
    p_r, r_r = percentile_curve(rewards_rarl)

    plt.figure(figsize=(6, 5))
    plt.plot(p_b, r_b, color='green', label=f'Baseline ({alg})')
    plt.plot(p_r, r_r, color='blue', label=f'RARL_{alg}')

    plt.xlabel('Percentile')
    plt.ylabel('Reward')
    plt.title(environment)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{alg}_vs_RARL_{alg}_percentile_comparison.png', dpi=300)
    plt.close()
    
def main():
    
    run_percentile_test(environment='Walker2d',alg='PPO', n_tests = 100)   
    
if __name__ == '__main__':

    main()