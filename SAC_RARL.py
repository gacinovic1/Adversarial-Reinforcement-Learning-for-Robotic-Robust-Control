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

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done =  zip(*batch)

        state = self.stack(state).squeeze(dim=1)
        action = self.stack(action).squeeze(dim=1)
        reward = self.stack(reward)
        next_state = self.stack(next_state).squeeze(dim=1)
        done = self.stack(done)
        
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

class SAC():

    distribution : str = 'Normal'

    def __init__(self,
                 net_V            : nn.Module,
                 net_V_target     : nn.Module,
                 net_Q1           : nn.Module,
                 net_Q2           : nn.Module,
                 net_pi           : nn.Module,
                 env            : gym.Env,
                 lr_V             : float = 1e-3,
                 lr_Q             : float = 1e-3,
                 lr_pi          : float = 1e-3,
                 gamma          : float = 0.99,
                 tau        : float = 1e-2,
                 log_alpha  : float = 0.0,
                 lr_alpha    : float = 3e-4,
                 epsilon        : float = 1e-6,
                 criticV_weight  : float = 0.5,
                 criticQ_weight  : float = 0.5,
                 print_flag     : bool  = True,
                 save_interval  : int   = 10,
                 capacity      : int   = 100_000,
                 freq_upd      : int   = 1000,
                 device=torch.device('cpu'),
                 name = 'model') -> None:

        self.NN_V = net_V
        self.NN_V_target = net_V_target
        self.NN_Q1 = net_Q1
        self.NN_Q2 = net_Q2
        self.NN_pi = net_pi
        self.env = env
        self.lr_V = lr_V
        self.lr_Q = lr_Q
        self.lr_pi = lr_pi
        self.gamma = gamma
        self.tau = tau
        self.epsilon = epsilon
        self.log_alpha = torch.tensor(log_alpha,requires_grad=True)
        self.alpha = self.log_alpha.exp()
        self.lr_alpha = lr_alpha
        self.criticV_weight = criticV_weight
        self.criticQ_weight = criticQ_weight
        self.device = device
        self.replay_buffer = ReplayBuffer(capacity)
        self.freq_upd = freq_upd 

        self.save_interval = save_interval
        self.model_name = name

        self.print_flag = print_flag

        self.NN_V.to(self.device)
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
        dist = Normal(0, 1)
                
        # compute new log prob using new actions
            
        z = dist.sample(mean.shape).to(self.device)
        new_actions =  torch.tanh(mean + std*z).to(self.device)

        new_log_prob = dist.log_prob(mean + std*z).sum(dim=-1, keepdim=True)       
        new_log_prob -= torch.log(1 - new_actions.pow(2) + self.epsilon).sum(dim=-1, keepdim=True)
        
        return new_actions, new_log_prob
            
        

    # method for run the evironment using state already converted in tensor
    def act(self, state) -> list[float]:
        
        mean, log_std, _ = self.NN_pi.forward(state)
        std = log_std.exp()
        dist = Normal(0, 1)
        
        z = dist.sample(mean.shape).to(self.device)
        actions = mean + std*z
        
        return actions



    # method for updating the actor/critic using SAC loss
    def SAC_update(self, mini_batch=64, epoch=10) -> None:

        for e in range(epoch): 
            
            state, action, reward, next_state, done = self.replay_buffer.sample(mini_batch)
            
            next_state = next_state.to(self.device)
            action     = action.to(self.device)
            reward     = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
            done       = torch.FloatTensor(done).unsqueeze(1).to(self.device)
            
            mean_actor_loss = 0
            mean_V_loss = 0
            mean_Q_loss = 0
            mean_loss = 0
            
            # predict state value
            V = self.NN_V(state)

            # predict actions and log_probs
            new_actions, new_log_prob = self.get_action(state)
           
            #  Get new Q value from the Q networks 
            new_Q = torch.min(self.NN_Q1(state, new_actions), self.NN_Q2(state, new_actions))

            # Compute the target value and value loss
            V_target =  (new_Q - self.alpha*new_log_prob)
            V_loss = nn.MSELoss()(V, V_target.detach())
            
            self.optim_V.zero_grad()
            V_loss.backward()
            self.optim_V.step()
                
            #  Predict Q1 and Q2 values
            Q1 = self.NN_Q1(state, action)         
            Q2 = self.NN_Q2(state, action)
            
            # predict target value for Q (Q_hat)
            target_value_next = self.NN_V_target(next_state) 
            Q_hat = (reward + (1-done) * self.gamma * target_value_next)
        
            # compute V, Q1, Q2 losses
            loss_Q1 = nn.MSELoss()(Q1, Q_hat.detach())
            loss_Q2 = nn.MSELoss()(Q2, Q_hat.detach())
            
            # backward
            self.optim_Q1.zero_grad()
            loss_Q1.backward()
            self.optim_Q1.step()
            
            self.optim_Q2.zero_grad()
            loss_Q2.backward()
            self.optim_Q2.step()
            
            # policy loss
            new_Q_prime = torch.min(self.NN_Q1(state, new_actions), self.NN_Q2(state, new_actions))
            self.alpha = self.log_alpha.exp()
            actor_loss  = (new_log_prob*self.alpha.detach() - new_Q_prime).mean()
            
            # policy backward
            self.optim_pi.zero_grad()
            actor_loss.backward()
            self.optim_pi.step()
            
            # Soft update V target net 
            for target_param, param in zip(self.NN_V_target.parameters(), self.NN_V.parameters()):

                target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
                
            # alpha loss
            with torch.no_grad():
                
                _ , new_log_prob1 = self.get_action(state)
            
            target_entropy = -1.0 * new_actions.size(-1) 
            alpha_loss = -(self.log_alpha * (new_log_prob1 + target_entropy).detach()).mean()
            
            self.optim_alpha.zero_grad()
            alpha_loss.backward()
            self.optim_alpha.step()
            
            # compute overall loss
            loss = actor_loss.item() + self.criticQ_weight*(loss_Q1.item() + loss_Q2.item()) + self.criticV_weight * V_loss.item() 

            mean_loss += loss
            mean_actor_loss += actor_loss.item()
            mean_V_loss += V_loss.item()
            mean_Q_loss += (loss_Q1.item() + loss_Q2.item())/2
            
            if self.print_flag: print(f"\t Iter: {(e+1)} | Loss: {(loss):>10.3f} | A: {(actor_loss.item()):>10.3f} | V: {(V_loss.item()):>10.3f} | Q: {((loss_Q1.item() + loss_Q2.item())/2):>10.3f} | ")


    # method for training the agent
    def train(self, episodes = 1024, epoch = 4, mini_batch = 128, max_steps_rollouts = 1024, continue_prev_train = False, brake_after_done = False) -> None:

        if(continue_prev_train):
            self.load()

        self.optim_V = torch.optim.Adam(self.NN_V.parameters(), lr = self.lr_V, maximize=False)
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
                
                # get the output of the actor     
                mean, log_std = self.NN_pi.forward(state.to(self.device))
                std = log_std.exp()

                # gaussian reparameterization trick
                dist = Normal(0, 1)
                z = dist.sample(mean.shape).to(self.device)
                action_pre =  mean + std*z
                action = torch.tanh(action_pre).detach()

                steps += 1
        
                # take a step in the environment
                next_state, reward, terminated, truncated, _ = self.env.step(action_pre)
                done = terminated or truncated
                
                self.replay_buffer.push(state, action, reward, next_state, done)
                
                if len(self.replay_buffer) > mini_batch and steps % self.freq_upd == 0:
                    
                    # starting the updating epoches
                    if self.print_flag: print("[T]: starting SAC iterations")
                    self.SAC_update(mini_batch, epoch)
                    if self.print_flag: print("[T]: end SAC iterations")
                    window += 1
                    
                state = next_state
                mean_rew += reward
               
                if self.print_flag:
                    print(' @', end = '')
                    print(f"\t reward: {mean_rew}")
                
                if done:
                    if self.print_flag: print("/", end='', flush=True)
                    tranches += 1
                    state, _ = self.env.reset()
                    continue
                
            print(f"[T]: end episode {episode} windows: | {window} | mean rewards: {mean_rew/tranches}")
                
            
            if(np.mod(episode, self.save_interval) == 0):
                self.save()    
            
        self.save()
        return


    def save(self) -> None:
        checkpoint = {
                        "model_pi": self.NN_pi.state_dict(),
                        "model_V": self.NN_V.state_dict(),
                        "model_Q1": self.NN_Q1.state_dict(),
                        "model_Q2": self.NN_Q2.state_dict(),
                        "model_V_target": self.NN_V_target.state_dict(),
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
        self.NN_V.load_state_dict(checkpoint["model_V"])
        self.NN_Q1.load_state_dict(checkpoint["model_Q1"])
        self.NN_Q2.load_state_dict(checkpoint["model_Q2"])
        self.NN_V_target.load_state_dict(checkpoint["model_V_target"])

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

    distribution : str = 'Beta'

    def __init__(self,
                 player         : nn.Module,
                 opponent       : nn.Module,
                 env            : gym.Env,
                 lr_player      : float = 1e-3,
                 lr_opponent   : float = 1e-3,
                 gamma          : float = 0.99,
                 lambda_        : float = 0.95,
                 clip           : float = 0.2,
                 critic_weight  : float = 0.5,
                 entropy_weight : float = 0.01,
                 print_flag     : bool  = True,
                 save_interval  : int   = 10,
                 device=torch.device('cpu'),
                 name = 'model') -> None:

        self.player = player
        self.opponent = opponent
        self.env = env
        self.lr_player = lr_player
        self.lr_opponent = lr_opponent
        self.gamma = gamma
        self.lambda_ = lambda_
        self.clip_param = clip
        self.critic_weight = critic_weight
        self.entropy_weight = entropy_weight
        self.device = device

        self.save_interval = save_interval
        self.model_name = name
        self.print_flag = print_flag

        self.player_dict   = {'actor' : self.player  , 'icon' : '[P]'}
        self.opponent_dict = {'actor' : self.opponent, 'icon' : '[O]'}

        self.schedule = [self.player_dict, self.opponent_dict]

        self.player.to(self.device)
        self.opponent.to(self.device)

    def _get_dist(self, par1, par2) -> torch.distributions:
        if self.distribution == 'Beta':
            dist = Beta(par1, par2)
        return dist

    def swap_actor(self) -> dict:
        self.schedule[0], self.schedule[1] = self.schedule[1], self.schedule[0]
        return self.schedule[1]

    # method for run the evironment using state already converted in tensor
    def act(self, state) -> list[float]:
        alpha, beta, _ = self.player.forward(state)
        dist = self._get_dist(alpha, beta)
        actions = dist.sample()

        return actions
    
    def act_both(self, state) -> tuple[list[float], list[float]]:
        alpha, beta, _ = self.player.forward(state)
        dist = self._get_dist(alpha, beta)
        actions_player = dist.sample()

        alpha, beta, _ = self.opponent.forward(state)
        dist = self._get_dist(alpha, beta)
        actions_opponent = dist.sample()

        return actions_player, actions_opponent

    # method for comute the GAE
    def compute_gae(self, last_v, rewards, values, dones) -> tuple[list[float], list[float]]:

        gae = 0
        returns, advantages = [], []

        for t in reversed(range(len(rewards))): # compute in backward
            delta = rewards[t] + (1 - dones[t]) * self.gamma * last_v - values[t] # d = r + gamma * V[t+1] * !done - value
            gae = delta + (1 - dones[t]) * self.gamma * self.lambda_ * gae # gae = d + gamma*lambda* !done *gae
            last_v = values[t]
            advantages.insert(0, gae)
            returns.insert(0, gae + values[t]) # return = gae + V

        return returns, advantages

    # method for perform the rollouts
    def rollOut(self, max_steps = 1000) -> tuple[list, list, dict, dict, float]:
        states, dones = [], []
        player_dict_t, opponent_dict_t = {}, {}
        player_dict_t  ['actions'], player_dict_t  ['rewards'], player_dict_t  ['values'], player_dict_t  ['log_probs'] = [], [], [], []
        opponent_dict_t['actions'], opponent_dict_t['rewards'], opponent_dict_t['values'], opponent_dict_t['log_probs'] = [], [], [], []

        done = False
        s, _ = self.env.reset()
        steps = 1
        terminated = False

        while steps <= max_steps: # perform iterations on the environemnt

            # get the output of the actor/critic and generate the distribution
            alpha_p, beta_p, V_p = self.player.forward(s.to(self.device))
            dist_p = self._get_dist(alpha_p, beta_p)

            alpha_o, beta_o, V_o = self.opponent.forward(s.to(self.device))
            dist_o = self._get_dist(alpha_o, beta_o)

            # compute the action and save it
            action_p = dist_p.sample()
            action_o = dist_o.sample()

            # compute log probs of the actions
            log_prob_p = dist_p.log_prob(action_p).sum()
            log_prob_o = dist_o.log_prob(action_o).sum()

            # save the trajectory
            states.append(s)

            player_dict_t  ['actions'].append(action_p)
            player_dict_t  ['log_probs'].append(log_prob_p)
            player_dict_t  ['values'].append(V_p.detach())

            opponent_dict_t['actions'].append(action_o)
            opponent_dict_t['log_probs'].append(log_prob_o)
            opponent_dict_t['values'].append(V_o.detach())


            # take a step in the environment
            s, reward, terminated, truncated, _ = self.env.step(action_p, action_o)

            # save the terminations and the rewards
            done = terminated or truncated

            dones.append(done)

            player_dict_t  ['rewards'].append( reward)
            opponent_dict_t['rewards'].append(-reward)

            if(self.print_flag and np.mod(steps, 50) == 0):
                print("#", end='', flush=True)

            if done:
                if self.print_flag: print("/", end='', flush=True)
                break

            steps += 1
        if terminated: player_dict_t['next_v'], opponent_dict_t['next_v'] = 0, 0
        else:
          _, _, next_v_p = self.player.forward(s.to(self.device))
          player_dict_t  ['next_v'] = next_v_p.detach()

          _, _, next_v_o = self.opponent.forward(s.to(self.device))
          opponent_dict_t['next_v'] = next_v_o.detach()

        if self.print_flag:
            print(' @', end = '')
            print(f"\t reward: {np.sum(player_dict_t['rewards'])}")
        return states, dones, player_dict_t, opponent_dict_t, steps

    # method to extract mini bach from the all trajectory
    def mini_bach_manager(self, states, actions, log_probs, returns, advantages, mini_bach = 16):

        lens = len(advantages)
        iters = lens//(mini_bach+1) + 1

        perm = np.random.permutation(lens)

        states     = [states[i]     for i in perm]
        actions    = [actions[i]    for i in perm]
        log_probs  = [log_probs[i]  for i in perm]
        returns    = [returns[i]    for i in perm]
        advantages = [advantages[i] for i in perm]

        adv_norm = torch.stack(advantages, dim=0).reshape(-1,1)
        if lens > 1:
          adv_norm = (adv_norm - adv_norm.mean(dim=0)) / (adv_norm.std(dim=0) + 1e-8)

        indices = [i for i in range(0, lens, mini_bach)]

        for _ in range(iters):
            index = np.random.choice(indices)
            indices.remove(index)

            slicer = slice(index, index + mini_bach)
            yield torch.cat(states[slicer],  dim=0).to(self.device),\
                  torch.cat(actions[slicer], dim=0).to(self.device),\
                  torch.stack(log_probs[slicer],  dim=0).reshape(-1,1).to(self.device),\
                  torch.stack(returns[slicer],    dim=0).reshape(-1,1).to(self.device),\
                  adv_norm[slicer].to(self.device)

    # method for update the actor/critic using SAC loss
    def SAC_update(self, actor, optimizer, states, actions, log_probs, returns, advantages, epoch=10, mini_bach=64) -> None:

        for e in range(epoch): # for each iteration of update
            if self.print_flag: print(f"\tSAC iteration: {e+1}", end='')

            mean_actor_loss = 0
            mean_critic_loss = 0
            mean_entropy_loss = 0
            mean_loss = 0
            iters = len(returns)//(mini_bach+1)+1

            # for each mini bach
            for state,  action,  log_prob,  return_, advantage in self.mini_bach_manager(
                states, actions, log_probs, returns, advantages, min(mini_bach, len(returns))):

                # extract the new value and the new disribution from actor/critic
                alpha, beta, V = actor.forward(state)

                dist = self._get_dist(alpha, beta)
                entropy = torch.clamp(dist.entropy().mean(), -10_000, 10_000) # compute entropy

                # compute new log prob using old actions
                new_log_prob = dist.log_prob(action).sum(dim=1)

                # compute the ratio of the log prob
                ratio = torch.exp(new_log_prob.reshape(-1,1) - log_prob)

                # compute the two item for the minimum on the loss
                a_loss = ratio * advantage
                a_loss_clip = ratio.clamp(min = 1 - self.clip_param, max = 1 + self.clip_param) * advantage # clipped

                # compute losses
                actor_loss  = -torch.min(a_loss, a_loss_clip).mean()
                critic_loss = torch.clamp(F.mse_loss(V.squeeze(1), return_.squeeze(1)), -10_000, 10_000)
                loss = actor_loss + self.critic_weight * critic_loss - self.entropy_weight * entropy

                mean_loss += loss
                mean_actor_loss += actor_loss
                mean_critic_loss += critic_loss
                mean_entropy_loss += entropy

                # backpropagate
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if self.print_flag: print(f"\t| Loss: {(mean_loss/iters):>10.3f} | A: {(mean_actor_loss/iters):>10.3f} | C: {(mean_critic_loss/iters):>10.3f} | E: {(mean_entropy_loss/iters):>10.3f} | i = {iters} |")

    # method for train the agent
    def train(self, player_episode = 2, opponent_episode = 1, episodes = 1024, epoch = 10, mini_bach = 128, max_steps_rollouts = 1024, continue_prev_train = False, brake_after_done = False) -> None:

        if(continue_prev_train):
            self.load()

        self.player_dict  ['optimizer'] = torch.optim.Adam(self.player.parameters()  , lr = self.lr_player  , maximize=False)
        self.opponent_dict['optimizer'] = torch.optim.Adam(self.opponent.parameters(), lr = self.lr_opponent, maximize=False)

        self.player_dict  ['max_episode'] = player_episode
        self.opponent_dict['max_episode'] = opponent_episode

        # repeat for the number of episodes
        for episode in range(episodes):
            
            if episode == 0 or current_actor_dict['episode_count'] >= current_actor_dict['max_episode']:
                current_actor_dict = self.swap_actor()
                current_actor_dict['episode_count'] = 0
        
            states_t = []
            self.player_dict  ['actions'], self.player_dict  ['log_probs'], self.player_dict  ['returns'], self.player_dict  ['advantages'] = [], [], [], []
            self.opponent_dict['actions'], self.opponent_dict['log_probs'], self.opponent_dict['returns'], self.opponent_dict['advantages'] = [], [], [], []

            steps = 0
            tranchs = 0
            mean_rew = 0
            print(f"{current_actor_dict['icon']}: starting roll-outs {episode}: ", end='')

            while steps <= max_steps_rollouts:

                # perform the rollout
                with torch.no_grad():
                    states, dones, self.player_dict['rollout_dict'], self.opponent_dict['rollout_dict'], steps_done = self.rollOut(max_steps_rollouts - steps)

                # compute the gae and take the return, compute the advantage only for the current actor
                with torch.no_grad():
                    current_actor_dict['rollout_dict']['returns'], current_actor_dict['rollout_dict']['advantages'] = self.compute_gae(current_actor_dict['rollout_dict']['next_v' ], 
                                                                                                       current_actor_dict['rollout_dict']['rewards'], 
                                                                                                       current_actor_dict['rollout_dict']['values' ], 
                                                                                                       dones)
                
                # update the dict for the curent actor
                states_t.extend(states)
                current_actor_dict['actions'   ].extend(current_actor_dict['rollout_dict']['actions'   ])
                current_actor_dict['log_probs' ].extend(current_actor_dict['rollout_dict']['log_probs' ])
                current_actor_dict['returns'   ].extend(current_actor_dict['rollout_dict']['returns'   ])
                current_actor_dict['advantages'].extend(current_actor_dict['rollout_dict']['advantages'])

                mean_rew += np.sum(self.player_dict['rollout_dict']['rewards'])
                steps += steps_done
                tranchs += 1
                if brake_after_done: break

            print(f" | {tranchs} | mean rewards: {mean_rew/tranchs}")
            if self.print_flag: print(f"{current_actor_dict['icon']}: end roll-outs {episode} ")

            # starting the iteration of PPO for the current actor
            if self.print_flag: print(f"{current_actor_dict['icon']}: starting PPO iterations")
            self.PPO_update(current_actor_dict['actor'],
                            current_actor_dict['optimizer'],
                            states_t, 
                            current_actor_dict['actions'], 
                            current_actor_dict['log_probs'], 
                            current_actor_dict['returns'], 
                            current_actor_dict['advantages'], 
                            epoch, mini_bach)
            
            if self.print_flag: print(f"{current_actor_dict['icon']}: end PPO iterations")

            if(np.mod(episode, self.save_interval) == 0):
                self.save()
            
            current_actor_dict['episode_count'] += 1

        self.state_queue = []
        self.save()
        return

    def save(self) -> None:
        checkpoint = {
            'player_state_dict'  : self.player.state_dict(),
            'opponent_state_dict': self.opponent.state_dict()
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

        self.player.load_state_dict(checkpoint['player_state_dict'])
        self.opponent.load_state_dict(checkpoint['opponent_state_dict'])

        if self.env.is_norm_wrapper:
            self.env.env.obs_rms.mean  = checkpoint['obs_mean']
            self.env.env.obs_rms.var   = checkpoint['obs_var']
            self.env.env.obs_rms.count = checkpoint['obs_count']

        print("** [LOADED OLD MODEL] **")

    def to(self, device):
        ret = super().to(device)
        ret.device = device
        return ret
    

    

    