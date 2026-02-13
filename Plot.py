import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

def plot_reward_vs_pert(csv_files,base_algs=('PPO', 'SAC'), rarl_algs=('RARL'),use_std=True,title='Walker2D', perturbation = ("Mass", "Friction")):
 

    if base_algs[0] == 'PPO' and (rarl_algs[0] == 'RARL' or rarl_algs[0] == 'RARL_PPO'):
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

   
    plt.xlabel(f'{pert}')
    plt.ylabel('Reward')
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(f'{base_algs[0]}_reward_vs_{pert}_{title}.png', dpi=300)
    plt.close()
    

def build_heatmap_data(csv_files):
    records = []

    for f in csv_files:
        
        df = pd.read_csv(f)
        records.append({'Mass': df['Mass'].iloc[0], 'Friction': df['Friction'].iloc[0], 'Reward': df['reward'].mean()})

    data = pd.DataFrame(records)

    pivot = data.pivot_table(index='Mass', columns='Friction', values='Reward', aggfunc='mean')
    pivot = pivot.sort_index(ascending=False)
    pivot = pivot.sort_index(axis=1)

    return pivot

def draw_heatmap(ax, pivot, title, cmap, vmin, vmax):
    
    im = ax.imshow(pivot.values, aspect='auto', cmap=cmap, interpolation='nearest', vmin=vmin, vmax=vmax) 

    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([f'{v:.2f}' for v in pivot.columns])

    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([f'{v:.1f}' for v in pivot.index])

    ax.set_xlabel('Friction coefficient')
    ax.set_ylabel('Mass')
    ax.set_title(title)

    return im


def plot_reward_heatmaps_ppo_vs_rarl(alg, base_path,cmap='RdBu_r', save_path=None, title='Walker2D'):
    
    base_path = Path(base_path)
    
    csvs_alg = [p for p in base_path.glob('*heatmap.csv') if alg in p.name and f'RARL_{alg}' not in p.name]
    csvs_rarl = [p for p in base_path.glob('*heatmap.csv') if alg in p.name and f'RARL_{alg}' in p.name]

    pivot_alg = build_heatmap_data(csvs_alg)
    pivot_rarl = build_heatmap_data(csvs_rarl)

    vmin = min(pivot_alg.min().min(), pivot_rarl.min().min())
    vmax = max(pivot_alg.max().max(), pivot_rarl.max().max())

    fig = plt.figure(figsize=(14, 7))

    gs = fig.add_gridspec(2, 2,height_ratios=[20, 1], hspace=0.3, wspace=0.15)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    cax = fig.add_subplot(gs[1, :])  

    im1 = draw_heatmap(ax1, pivot_alg, alg, cmap, vmin, vmax)
    im2 = draw_heatmap(ax2, pivot_rarl, f'RARL_{alg}', cmap, vmin, vmax)

    cbar = fig.colorbar(im1, cax=cax, orientation='horizontal')
    cbar.set_label('Reward')

   # plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.savefig(f'{alg}_vs_RARL_{title}_heatmaps.png', dpi=300)
        plt.close()
    
    
def main(heatmap = False, env = "Walker2D"):
    
    if not heatmap:
    
        alg = "SAC" # 'PPO', 'RARL_PPO', 'SAC', 'RARL_SAC'
        path = Path('./Files/'f'{env}/heatmap')
        csvs_mass = [p for p in path.glob('*.csv') if alg in p.name and (('Mass' in p.name) and not('Friction' in p.name)) or (not('Mass' in p.name) and not('Friction' in p.name))] 
        csvs_frict = [p for p in path.glob('*.csv') if alg in p.name and (not('Mass' in p.name) and ('Friction' in p.name)) or (not('Mass' in p.name) and not('Friction' in p.name))]
        
        plot_reward_vs_pert(
            csv_files=csvs_mass,
            base_algs=("SAC",),
            rarl_algs=("RARL_SAC",),
            use_std=True,  
            title=env,
            perturbation = ('Mass',),
        )
        plot_reward_vs_pert(
            csv_files=csvs_frict,
            base_algs=("SAC",),
            rarl_algs=("RARL_SAC",),
            use_std=True,  
            title=env,
            perturbation = ('Friction',),
        )
        
    else:
        
        BASE = Path('./Files/'f'{env}/heatmap')
        alg = "SAC"  #  or "SAC" 

        plot_reward_heatmaps_ppo_vs_rarl(alg = alg,base_path=BASE, title=env)
    
def xor(a, b):
    return (a and not b) or (not a and b)

if __name__ == '__main__':

    env = "Swimmer" # "Walker2D"
    main(heatmap=False, env = env)
    main(heatmap=True, env = env)