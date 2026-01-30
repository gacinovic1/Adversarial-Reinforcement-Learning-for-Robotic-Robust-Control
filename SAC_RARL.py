import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from torch.distributions import Beta, Normal

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def stack(self, items):
        
        first = items[0]
        if isinstance(first, torch.Tensor):
            return torch.stack(items)
        elif isinstance(first, np.ndarray):
            return np.stack(items)
        else:
            return np.array(items)

    def push(self, state, action, reward, next_state, mask):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, mask)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, mask =  zip(*batch)

        state = self.stack(state).squeeze(dim=1)
        action = self.stack(action).squeeze(dim=1)
        reward = self.stack(reward)
        next_state = self.stack(next_state).squeeze(dim=1)
        mask = self.stack(mask)
        
        return state, action, reward, next_state, mask
    
    def __len__(self):
        return len(self.buffer)


class SAC():

    distribution : str = 'Normal'

    def __init__(self,
                 net_Q1_target     : nn.Module,
                 net_Q2_target    : nn.Module,
                 net_Q1           : nn.Module,
                 net_Q2           : nn.Module,
                 net_pi           : nn.Module,
                 env            : gym.Env,
                 lr_Q             : float = 1e-4,
                 lr_pi          : float = 1e-4,
                 gamma          : float = 0.99,
                 tau        : float = 0.005,
                 log_alpha  : float = -1.60944,  # to start with alpha = 0.2
                 lr_alpha    : float = 1e-4,
                 epsilon        : float = 1e-6,
                 print_flag     : bool  = True,
                 save_interval  : int   = 10,
                 start_policy   : int   = 0,
                 capacity      : int   = 1_000_000,
                 freq_upd      : int   = 1,
                 freq_ep       : int   = 1,
                 device=torch.device('cpu'),
                 name = 'model') -> None:

        self.NN_Q1_target = net_Q1_target
        self.NN_Q2_target = net_Q2_target
        self.NN_Q1 = net_Q1
        self.NN_Q2 = net_Q2
        self.NN_pi = net_pi
        self.env = env
        self.lr_Q = lr_Q
        self.lr_pi = lr_pi
        self.gamma = gamma
        self.tau = tau
        self.epsilon = epsilon
        self.log_alpha = torch.tensor(log_alpha,requires_grad=True)
        self.alpha = self.log_alpha.exp()
        self.lr_alpha = lr_alpha
        self.device = device
        self.replay_buffer = ReplayBuffer(capacity)
        self.freq_upd = freq_upd 
        self.start_policy = start_policy
        self.freq_ep = freq_ep

        self.save_interval = save_interval
        self.model_name = name

        self.print_flag = print_flag

        self.NN_Q1.to(self.device)
        self.NN_Q2.to(self.device)
        self.NN_pi.to(self.device)

    def _get_dist(self, par1, par2) -> torch.distributions:
        if self.distribution == 'Normal':
            dist = Normal(par1, par2)
        return dist
    
    def get_action(self, state):
        
        # extract the new disribution from actor
        mean, log_std = self.NN_pi.forward(state)
        std = log_std.exp()
        
        dist = Normal(mean, std)
        
        action_pre = dist.rsample()  # equivalent to (mean + std * N(0,1))
        action = torch.tanh(action_pre)

        new_log_prob = dist.log_prob(action_pre).sum(dim=-1, keepdim=True)       
        new_log_prob -= torch.log((1-action.pow(2)) + self.epsilon).sum(dim=-1, keepdim=True)
        
        return action, action_pre, new_log_prob
            
        

    # method for acting the environment using state already converted in tensor
    def act(self, state) -> list[float]:

        mean, log_std = self.NN_pi.forward(state)
        std = log_std.exp()
        
        dist = Normal(mean, std)
        action_pre = dist.rsample() 
        action = torch.tanh(action_pre) 
        
        return action

    # method for updating the actor/critic using SAC loss
    def SAC_update(self, mini_batch=64, epoch=10) -> None:

        for e in range(epoch): 
            
            state, action, reward, next_state, mask = self.replay_buffer.sample(mini_batch)
            
            state      = state.to(self.device)
            next_state = next_state.to(self.device)
            action     = action.to(self.device)
            reward     = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
            mask       = torch.FloatTensor(mask).unsqueeze(1).to(self.device)
            
            mean_actor_loss = 0
            mean_Q_loss = 0
            mean_loss = 0
            
            with torch.no_grad():
                
                # predict actions and log_probs
                new_actions, _ , new_log_probs = self.get_action(next_state)
                
                # predict target for Q

                Q1_target, Q2_target =  self.NN_Q1_target(next_state, new_actions), self.NN_Q2_target(next_state, new_actions)
                Q_target = torch.min(Q1_target, Q2_target) - self.alpha * new_log_probs
                Q_hat = (reward + mask * self.gamma * Q_target)
                
            #  Predict Q1 and Q2 values
            Q1 = self.NN_Q1(state, action)         
            Q2 = self.NN_Q2(state, action)
            
            # compute  Q1, Q2 losses
            loss_Q1 = F.mse_loss(Q1, Q_hat)  # try to invert it!
            loss_Q2 = F.mse_loss(Q2, Q_hat)
            
            # backward
            self.optim_Q1.zero_grad()
            loss_Q1.backward()
            self.optim_Q1.step()
            
            self.optim_Q2.zero_grad()
            loss_Q2.backward()
            self.optim_Q2.step()

            new_action, _ , new_log_prob = self.get_action(state)

            # policy loss
            new_Q_prime = torch.min(self.NN_Q1(state, new_action), self.NN_Q2(state, new_action))
            actor_loss  = ((self.alpha*new_log_prob) - new_Q_prime).mean()   # try to invert it!
            
            # policy backward
            self.optim_pi.zero_grad()
            actor_loss.backward()
            self.optim_pi.step()
            
            # alpha loss
        
            target_entropy = -1.0 * new_action.size(-1)   # target entropy is -|A|
            target_entropy = torch.tensor(target_entropy).to(self.device).item()
            alpha_loss = -(self.log_alpha * (new_log_prob + target_entropy).detach()).mean()
            
            self.optim_alpha.zero_grad()
            alpha_loss.backward()
            self.optim_alpha.step()
            
            # update alpha
            self.alpha = self.log_alpha.exp()
            
            # Soft update Q1 and Q2 target nets 
            for target_param, param in zip(self.NN_Q1_target.parameters(), self.NN_Q1.parameters()):

                target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau) # try to invert it! 
            
            for target_param, param in zip(self.NN_Q2_target.parameters(), self.NN_Q2.parameters()):

                target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
                
            
            # compute overall loss
            loss = actor_loss.item() +(loss_Q1.item() + loss_Q2.item())/2 

            mean_loss += loss
            mean_actor_loss += actor_loss.item()
            mean_Q_loss += (loss_Q1.item() + loss_Q2.item())/2
            
            if self.print_flag: print(f"\t Iter: {(e+1)} | Loss: {(loss):>10.3f} | A: {(actor_loss.item()):>10.3f} | Q: {((loss_Q1.item() + loss_Q2.item())/2):>10.3f} | ")


    # method for training the agent
    def train(self, episodes = 1024, epoch = 4, mini_batch = 128, max_steps_rollouts = 1024, continue_prev_train = False) -> None:

        if(continue_prev_train):
            self.load()

        self.optim_Q1 = torch.optim.Adam(self.NN_Q1.parameters(), lr = self.lr_Q, maximize=False)
        self.optim_Q2 = torch.optim.Adam(self.NN_Q2.parameters(), lr = self.lr_Q, maximize=False)
        self.optim_pi = torch.optim.Adam(self.NN_pi.parameters(), lr = self.lr_pi, maximize=False)
        self.optim_alpha = torch.optim.Adam([self.log_alpha], lr = self.lr_alpha, maximize=False)

        # repeat for the number of episodes
        for episode in range(episodes):

            done = False
            state, _ = self.env.reset()
            steps = 1
            terminated = False
            mean_rew = 0
            window = 0
            tranches = 0

            while steps <= max_steps_rollouts: # perform iterations on the environemnt
                
                if episode < self.start_policy:
                    # sample random action
                    action = self.env.env.action_space.sample() 
                    action = torch.tensor(action, dtype=torch.float32).detach()
                    action = action.unsqueeze(0)
                    
                else:
                    # get the output of the actor 
                    action, _, _ = self.get_action(state)
                    action = action.detach()
        
                # take a step in the environment
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                # Ignore the "done" signal if it comes from hitting the time horizon.
                #mask = 1.0 if steps >= max_steps_rollouts else float(not done)
                mask = 0.0 if terminated else 1.0
                
                self.replay_buffer.push(state, action, reward, next_state, mask)
                
                if len(self.replay_buffer) > mini_batch and steps % self.freq_upd == 0 and episode % self.freq_ep == 0:
                    
                    # starting the updating epoches
                    if self.print_flag: print("[T]: starting SAC iterations")
                    self.SAC_update(mini_batch, epoch)
                    if self.print_flag: print("[T]: end SAC iterations")
                    window += 1
                    
                steps += 1
                    
                state = next_state
                mean_rew += reward
               
                if self.print_flag:
                    print(' @', end = '')
                    print(f"\t reward: {mean_rew}")
                
                if done:
                    if self.print_flag: print("/", end='', flush=True)
                    tranches += 1
                    state, _ = self.env.reset()
                    done = False
                    continue
                
            print(f"[T]: end episode {episode} windows: | {window} | tranches: {tranches} | mean rewards: {mean_rew/tranches} ")
                
            
            if(np.mod(episode, self.save_interval) == 0):
                self.save()    
            
        self.save()
        return


    def save(self) -> None:
        checkpoint = {
                        "model_pi": self.NN_pi.state_dict(),
                        "model_Q1": self.NN_Q1.state_dict(),
                        "model_Q2": self.NN_Q2.state_dict(),
                        "model_Q1_target": self.NN_Q1_target.state_dict(),
                        "model_Q2_target": self.NN_Q1_target.state_dict(),
                        "Replay_buffer": self.replay_buffer
                    }

        if self.env.is_norm_wrapper:
            checkpoint['obs_mean']  = self.env.env.obs_rms.mean
            checkpoint['obs_var']   = self.env.env.obs_rms.var
            checkpoint['obs_count'] = self.env.env.obs_rms.count

        torch.save(checkpoint, self.model_name + '.pt')
        print("** [PROGRES SAVED] **")

    def load(self, model_name = None) -> None:
        if model_name == None:
            model_name = self.model_name + '.pt'

        checkpoint = torch.load(model_name, map_location=self.device, weights_only=False)
    
        self.NN_pi.load_state_dict(checkpoint["model_pi"])
        self.NN_Q1.load_state_dict(checkpoint["model_Q1"])
        self.NN_Q2.load_state_dict(checkpoint["model_Q2"])
        self.NN_Q1_target.load_state_dict(checkpoint["model_Q1_target"])
        self.NN_Q2_target.load_state_dict(checkpoint["model_Q1_target"])
        self.replay_buffer = checkpoint["Replay_buffer"]

        if self.env.is_norm_wrapper:
            self.env.env.obs_rms.mean  = checkpoint['obs_mean']
            self.env.env.obs_rms.var   = checkpoint['obs_var']
            self.env.env.obs_rms.count = checkpoint['obs_count']

        print("** [LOADED OLD MODEL] **")


    def to(self, device):
        ret = super().to(device)
        ret.device = device
        return ret



class RARL_SAC():

    distribution : str = 'Normal'

    def __init__(self,
                 player         : nn.Module,
                 opponent       : nn.Module,
                 env            : gym.Env,
                 lr_Q             : float = 1e-3,
                 lr_pi          : float = 1e-3,
                 gamma          : float = 0.99,
                 tau        : float = 0.005,
                 log_alpha  : float = -1.60944,  # to start with alpha = 0.2
                 lr_alpha    : float = 1e-4,
                 epsilon        : float = 1e-6,
                 print_flag     : bool  = True,
                 save_interval  : int   = 10,
                 start_policy   : int   = 0,
                 capacity      : int   = 1_000_000,
                 freq_upd      : int   = 1,
                 freq_ep       : int   = 1,
                 device=torch.device('cpu'),
                 name = 'model') -> None:


        self.player = player
        self.opponent = opponent
        self.env = env

        self.lr_Q = lr_Q
        self.lr_pi = lr_pi
        self.gamma = gamma
        self.tau = tau
        self.epsilon = epsilon
        self.log_alpha_player = torch.tensor(log_alpha,requires_grad=True)
        self.alpha_player = self.log_alpha_player.exp()
        self.log_alpha_opponent = torch.tensor(log_alpha,requires_grad=True)
        self.alpha_opponent = self.log_alpha_opponent.exp()
        self.lr_alpha = lr_alpha
        self.device = device
        self.replay_buffer_player = ReplayBuffer(capacity)
        self.replay_buffer_opponent = ReplayBuffer(capacity)
        self.freq_upd = freq_upd 
        self.start_policy = start_policy
        self.freq_ep = freq_ep

        self.save_interval = save_interval
        self.model_name = name

        self.print_flag = print_flag

        self.player_dict   = {'actor' : self.player  , 'icon' : '[P]', 'replay_buffer': self.replay_buffer_player, "log_alpha" : self.log_alpha_player, "alpha" : self.alpha_player}
        self.opponent_dict = {'actor' : self.opponent, 'icon' : '[O]','replay_buffer': self.replay_buffer_opponent, "log_alpha" : self.log_alpha_opponent, "alpha" : self.alpha_opponent}

        self.schedule = [self.player_dict, self.opponent_dict]

        self.player['policy'].to(self.device)
        self.player['Q1'].to(self.device)
        self.player['Q2'].to(self.device)
        self.player['Q1_target'].to(self.device)
        self.player['Q2_target'].to(self.device)
        
        self.opponent['policy'].to(self.device)
        self.opponent['Q1'].to(self.device)
        self.opponent['Q2'].to(self.device)
        self.opponent['Q1_target'].to(self.device)
        self.opponent['Q2_target'].to(self.device)

    def _get_dist(self, par1, par2) -> torch.distributions:
        if self.distribution == 'Normal':
            dist = Beta(par1, par2)
        return dist

    def swap_actor(self) -> dict:
        self.schedule[0], self.schedule[1] = self.schedule[1], self.schedule[0]
        return self.schedule[1]

    # method for run the evironment using state already converted in tensor
    def act(self, state) -> list[float]:
        
        mean, log_std = self.player['policy'].forward(state)
        std = log_std.exp()
        
        dist = Normal(mean, std)
        action_pre = dist.rsample() 
        action = torch.tanh(action_pre) 
        
        return action
    
    def act_both(self, state) -> tuple[list[float], list[float]]:
        
        mean, log_std = self.player['policy'].forward(state)
        std = log_std.exp()
        
        dist = Normal(mean, std)
        action_pre = dist.rsample() 
        actions_player = torch.tanh(action_pre) 

        mean, log_std = self.opponent['policy'].forward(state)
        std = log_std.exp()
        
        dist = Normal(mean, std)
        action_pre = dist.rsample() 
        actions_opponent = torch.tanh(action_pre) 

        return actions_player, actions_opponent

    def get_action(self, state, actor):
        
        # extract the new disribution from actor
        mean, log_std = actor["policy"].forward(state)
        std = log_std.exp()
        
        dist = Normal(mean, std)
        
        action_pre = dist.rsample()  # equivalent to (mean + std * N(0,1))
        action = torch.tanh(action_pre)

        new_log_prob = dist.log_prob(action_pre).sum(dim=-1, keepdim=True)       
        new_log_prob -= torch.log((1-action.pow(2)) + self.epsilon).sum(dim=-1, keepdim=True)
        
        return action, action, new_log_prob
            

       # method for updating the actor/critic using SAC loss
    def SAC_update(self, actor_dict, mini_batch=64, epoch=10) -> None:

        for e in range(epoch): 
            
            state, action, reward, next_state, mask = actor_dict['replay_buffer'].sample(mini_batch)
            
            state      = state.to(self.device)
            next_state = next_state.to(self.device)
            action     = action.to(self.device)
            reward     = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
            mask       = torch.FloatTensor(mask).unsqueeze(1).to(self.device)
            
            mean_actor_loss = 0
            mean_Q_loss = 0
            mean_loss = 0
            
            with torch.no_grad():
                
                # predict actions and log_probs
                new_actions, _ , new_log_probs = self.get_action(next_state, actor_dict['actor'])
                
                # predict target for Q

                Q1_target, Q2_target =  actor_dict['actor']['Q1_target'](next_state, new_actions), actor_dict['actor']['Q2_target'](next_state, new_actions)
                Q_target = torch.min(Q1_target, Q2_target) - actor_dict['alpha'] * new_log_probs
                Q_hat = (reward + mask * self.gamma * Q_target)
                
            #  Predict Q1 and Q2 values
            Q1 = actor_dict['actor']['Q1'](state, action)         
            Q2 = actor_dict['actor']['Q2'](state, action)
            
            # compute  Q1, Q2 losses
            loss_Q1 = F.mse_loss(Q1, Q_hat)  # try to invert it!
            loss_Q2 = F.mse_loss(Q2, Q_hat)
            
            # backward
            actor_dict['optimizer_Q1'].zero_grad()
            loss_Q1.backward()
            actor_dict['optimizer_Q1'].step()
            
            actor_dict['optimizer_Q2'].zero_grad()
            loss_Q2.backward()
            actor_dict['optimizer_Q2'].step()

            new_action, _ , new_log_prob = self.get_action(state, actor_dict['actor'])

            # policy loss
            new_Q_prime = torch.min(actor_dict['actor']['Q1'](state, new_action), actor_dict['actor']['Q2'](state, new_action))
            actor_loss  = ((actor_dict['alpha']*new_log_prob) - new_Q_prime).mean()   # try to invert it!
            
            # policy backward
            actor_dict['optimizer_pi'].zero_grad()
            actor_loss.backward()
            actor_dict['optimizer_pi'].step()
            
            # alpha loss
        
            target_entropy = -1.0 * new_action.size(-1)   # target entropy is -|A|
            target_entropy = torch.tensor(target_entropy).to(self.device).item()
            alpha_loss = -(actor_dict['log_alpha'] * (new_log_prob + target_entropy).detach()).mean()
            
            actor_dict['optimizer_alpha'].zero_grad()
            alpha_loss.backward()
            actor_dict['optimizer_alpha'].step()
            
            # update alpha
            actor_dict['alpha'] = actor_dict['log_alpha'].exp()
            
            # Soft update Q1 and Q2 target nets 
            for target_param, param in zip(actor_dict['actor']['Q1_target'].parameters(), actor_dict['actor']['Q1'].parameters()):

                target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau) # try to invert it! 
            
            for target_param, param in zip(actor_dict['actor']['Q2_target'].parameters(), actor_dict['actor']['Q2'].parameters()):

                target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
                
            
            # compute overall loss
            loss = actor_loss.item() +(loss_Q1.item() + loss_Q2.item())/2 

            mean_loss += loss
            mean_actor_loss += actor_loss.item()
            mean_Q_loss += (loss_Q1.item() + loss_Q2.item())/2
            
            if self.print_flag: print(f"\t Iter: {(e+1)} | Loss: {(loss):>10.3f} | A: {(actor_loss.item()):>10.3f} | Q: {((loss_Q1.item() + loss_Q2.item())/2):>10.3f} | ")


    # method for train the agent
    def train(self, player_episode = 2, opponent_episode = 1, episodes = 1024, epoch = 10, mini_batch = 128, max_steps_rollouts = 1024, continue_prev_train = False, brake_after_done = False) -> None:

        if(continue_prev_train):
            self.load()

        self.player_dict ['optimizer_Q1'] = torch.optim.Adam(self.player["Q1"].parameters(), lr = self.lr_Q, maximize=False)
        self.player_dict ['optimizer_Q2'] = torch.optim.Adam(self.player["Q2"].parameters(), lr = self.lr_Q, maximize=False)
        self.player_dict ['optimizer_pi']= torch.optim.Adam(self.player["policy"].parameters(), lr = self.lr_pi, maximize=False)
        self.player_dict ['optimizer_alpha'] = torch.optim.Adam([self.log_alpha_player], lr = self.lr_alpha, maximize=False)
        
        self.opponent_dict ['optimizer_Q1'] = torch.optim.Adam(self.opponent["Q1"].parameters(), lr = self.lr_Q, maximize=False)
        self.opponent_dict ['optimizer_Q2'] = torch.optim.Adam(self.opponent["Q2"].parameters(), lr = self.lr_Q, maximize=False)
        self.opponent_dict ['optimizer_pi']= torch.optim.Adam(self.opponent["policy"].parameters(), lr = self.lr_pi, maximize=False)
        self.opponent_dict ['optimizer_alpha'] = torch.optim.Adam([self.log_alpha_opponent], lr = self.lr_alpha, maximize=False)

        self.player_dict  ['max_episode'] = player_episode
        self.opponent_dict['max_episode'] = opponent_episode

        # repeat for the number of episodes
        for episode in range(episodes):
            
            if episode == 0 or current_actor_dict['episode_count'] >= current_actor_dict['max_episode']:
                current_actor_dict = self.swap_actor()
                current_actor_dict['episode_count'] = 0
                print("\nStarting " + str(current_actor_dict['icon'])+" training phase\n" )
                
            done = False
            state, _ = self.env.reset()
            steps = 1
            terminated = False
            mean_rew = 0
            window = 0
            tranches = 0
            
            while steps <= max_steps_rollouts: # perform iterations on the environemnt
                
                if episode < self.start_policy:
                    # sample random action for player
                    action_player = self.env.env.action_space.sample() 
                    action_player = torch.tensor(action_player, dtype=torch.float32).detach()
                    action_player = action_player.unsqueeze(0)
                    action_opponent, _, _ = self.get_action(state, self.opponent)
                    action_opponent = action_opponent.detach()
                    
                else:
                    # get the output of the actor 
                    action_player, _, _ = self.get_action(state, self.player)
                    action_opponent, _, _ = self.get_action(state, self.opponent)
                    action_player = action_player.detach()
                    action_opponent = action_opponent.detach()
        
                # take a step in the environment
                next_state, reward, terminated, truncated, _ = self.env.step(action_player, action_opponent)
                done = terminated or truncated
                
                if current_actor_dict['actor'] == self.player:
                    action = action_player
                else:
                    action = action_opponent
                    reward = -reward
                
                # Ignore the "done" signal if it comes from hitting the time horizon.
                mask = 0.0 if terminated else 1.0
                
                current_actor_dict['replay_buffer'].push(state, action, reward, next_state, mask)
                
                if len(current_actor_dict['replay_buffer']) > mini_batch and steps % self.freq_upd == 0 and episode % self.freq_ep == 0:
                    
                    # starting the updating epoches
                    if self.print_flag: print("[T]: starting SAC iterations")
                    self.SAC_update( current_actor_dict, mini_batch, epoch)
                    if self.print_flag: print("[T]: end SAC iterations")
                    window += 1
                    
                steps += 1
                    
                state = next_state
                mean_rew += reward
               
                if self.print_flag:
                    print(' @', end = '')
                    print(f"\t reward: {mean_rew}")
                
                if done:
                    if self.print_flag: print("/", end='', flush=True)
                    tranches += 1
                    state, _ = self.env.reset()
                    done = False
                    continue
                
            print(f"[T]: end episode {episode} windows: | {window} | mean rewards: {mean_rew/tranches}")
                
            if(np.mod(episode, self.save_interval) == 0):
                self.save()
                
            current_actor_dict['episode_count'] += 1    
            
        self.save()
        return


    def save(self) -> None:
        checkpoint = {
            'player_state_dict'  : self.player['policy'].state_dict(),
            'player_Q1_state_dict'  : self.player['Q1'].state_dict(),
            'player_Q2_state_dict'  : self.player['Q2'].state_dict(),
            'player_Q1_target_state_dict'  : self.player['Q1_target'].state_dict(),
            'player_Q2_target_state_dict'  : self.player['Q2_target'].state_dict(),
            
            'opponent_state_dict': self.opponent['policy'].state_dict(),
            'opponent_Q1_state_dict': self.opponent['Q1'].state_dict(),
            'opponent_Q2_state_dict': self.opponent['Q2'].state_dict(),
            'opponent_Q1_target_state_dict': self.opponent['Q1_target'].state_dict(),
            'opponent_Q2_target_state_dict': self.opponent['Q2_target'].state_dict()
            }

        if self.env.is_norm_wrapper:
            checkpoint['obs_mean']  = self.env.env.obs_rms.mean
            checkpoint['obs_var']   = self.env.env.obs_rms.var
            checkpoint['obs_count'] = self.env.env.obs_rms.count

        torch.save(checkpoint, self.model_name + '.pt')
        print("** [PROGRES SAVED] **")

    def load(self, model_name = None) -> None:
        if model_name == None:
            model_name = self.model_name + '.pt'

        checkpoint = torch.load(model_name, map_location=self.device, weights_only=False)
        
        self.player['policy'].load_state_dict(checkpoint['player_state_dict'])
        self.player['Q1'].load_state_dict(checkpoint['player_Q1_state_dict'])
        self.player['Q2'].load_state_dict(checkpoint['player_Q2_state_dict'])
        self.player['Q1_target'].load_state_dict(checkpoint['player_Q1_target_state_dict'])
        self.player['Q2_target'].load_state_dict(checkpoint['player_Q2_target_state_dict'])
        
        self.opponent['policy'].load_state_dict(checkpoint['opponent_state_dict'])
        self.opponent['Q1'].load_state_dict(checkpoint['opponent_Q1_state_dict'])
        self.opponent['Q2'].load_state_dict(checkpoint['opponent_Q2_state_dict'])
        self.opponent['Q1_target'].load_state_dict(checkpoint['opponent_Q1_target_state_dict'])
        self.opponent['Q2_target'].load_state_dict(checkpoint['opponent_Q2_target_state_dict'])
        if self.env.is_norm_wrapper:
            self.env.env.obs_rms.mean  = checkpoint['obs_mean']
            self.env.env.obs_rms.var   = checkpoint['obs_var']
            self.env.env.obs_rms.count = checkpoint['obs_count']

        print("** [LOADED OLD MODEL] **")

    def to(self, device):
        ret = super().to(device)
        ret.device = device
        return ret
    

    

    