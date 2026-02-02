import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

def plot_reward_vs_pert(csv_files,base_algs=('PPO', 'SAC'), rarl_algs=('RARL'),use_std=True,title='Walker2D', perturbation = ("Mass", "Friction")):
 

    if base_algs[0] == 'PPO' and rarl_algs[0] == 'RARL_PPO':
        label1 = "PPO"
        label2 = "RARL_PPO"
    elif base_algs[0] == 'SAC' and rarl_algs[0] == 'RARL_SAC':
        label1 = "SAC"
        label2 = "RARL_SAC"
    else:
        raise ValueError("base_algs and rarl_algs must be either ('PPO', 'RARL_PPO') or ('SAC', 'RARL_SAC')")

    df = pd.concat(pd.read_csv(f) for f in csv_files)

    df_base = df[df['algorithm'].isin(base_algs)]
    df_rarl = df[df['algorithm'].isin(rarl_algs)]
    pert = perturbation[0]


    def aggregate(data):
        
        g = data.groupby(pert)['reward']
        stats = pd.DataFrame({'mean': g.mean(),'std': g.std(),'min': g.min(),'max': g.max()}).reset_index()
        return stats.sort_values(pert)

    base_stats = aggregate(df_base)
    rarl_stats = aggregate(df_rarl)

    plt.figure(figsize=(8, 5))

    # ppo or sac
    plt.plot(base_stats[pert],base_stats['mean'],linewidth=3,label=label1)

    if use_std:
        plt.fill_between(base_stats[pert],base_stats['mean'] - base_stats['std'],base_stats['mean'] + base_stats['std'],alpha=0.3)
    else:
        plt.fill_between(base_stats[pert],base_stats['min'],base_stats['max'],alpha=0.3)

    # RARL 
    plt.plot(rarl_stats[pert],rarl_stats['mean'],linewidth=3,label=label2)

    if use_std:
        plt.fill_between(rarl_stats[pert],rarl_stats['mean'] - rarl_stats['std'],rarl_stats['mean'] + rarl_stats['std'],alpha=0.3)
    else:
        plt.fill_between(rarl_stats[pert],rarl_stats['min'],rarl_stats['max'],alpha=0.3)

   
    plt.xlabel(f'Perturbation of {pert} of the torso')
    plt.ylabel('Reward')
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(f'{title}_reward_vs_{pert}.png', dpi=300)
    plt.close()
    

def plot_reward_heatmap(csv_files,title='RARL_PPO',save_path=None,cmap='RdBu_r'):


    records = []

    for f in csv_files:
        
        df = pd.read_csv(f)
        mass = df['Mass'].iloc[0]
        friction = df['Friction'].iloc[0]
        reward_mean = df['reward'].mean()

        records.append({'Mass': mass,'Friction': friction,'Reward': reward_mean})

    data = pd.DataFrame(records)

    pivot = data.pivot(index='Mass', columns='Friction',values='Reward')

   
    pivot = pivot.sort_index(ascending=False)
    pivot = pivot.sort_index(axis=1)

    plt.figure(figsize=(6, 5))

    im = plt.imshow(pivot.values,aspect='auto',cmap=cmap,interpolation='nearest')

    plt.colorbar(im, label='Reward')
    plt.xticks(ticks=np.arange(len(pivot.columns)),labels=[f'{v:.2f}' for v in pivot.columns])
    plt.yticks(ticks=np.arange(len(pivot.index)),labels=[f'{v:.1f}' for v in pivot.index])
    plt.xlabel('Friction coefficient')
    plt.ylabel('Mass of torso')
    plt.title(title)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.savefig(f'{title}_reward_vs_both.png', dpi=300)
        plt.close()
    
    
def main(heatmap = False):
    
    if not heatmap:
    
        csvs = Path('/home/gacinovic/adversarial_RL/Adversarial-Reinforcement-Learning-for-Robotic-Robust-Control/Files/Walker2D').glob('*.csv')

        plot_reward_vs_pert(
            csv_files=csvs,
            base_algs=("PPO",),
            rarl_algs=("RARL_PPO",),
            use_std=False,  
            title='Walker2D',
            perturbation = ('Friction',),
        )
        
    else:
        
        BASE = Path('/home/gacinovic/adversarial_RL/Adversarial-Reinforcement-Learning-for-Robotic-Robust-Control/Files/Walker2D/heatmap')
        required = ['Walker_model_colab']
        csvs = [p for p in BASE.glob('*heatmap.csv')if all(k in p.name for k in required)]
        
        #csvs = Path('/home/gacinovic/adversarial_RL/Adversarial-Reinforcement-Learning-for-Robotic-Robust-Control/Files/Walker2D/heatmap').glob('*heatmap.csv')

        plot_reward_heatmap(
            csv_files=csvs,
            title='PPO'
            #save_path='Plots/rarl_mass_friction_heatmap.png'
        )
    
if __name__ == '__main__':

    main(heatmap=True)