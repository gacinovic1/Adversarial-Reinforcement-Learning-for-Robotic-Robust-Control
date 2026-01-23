import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ENV_wrapper(gym.Wrapper):

    HUMAN_RENDER = 'human'
    
    def __init__(self, env_name : str,
                        act_min : float, 
                        act_max : float, 
                    render_mode : str    = None, 
                is_norm_wrapper : bool   = True, algorithm : str = 'SAC'
                        ) -> None:

        self.env_name = env_name
        self.is_norm_wrapper = is_norm_wrapper
        self.algorithm = algorithm

        if render_mode == ENV_wrapper.HUMAN_RENDER:
            self.env = gym.make(self.env_name, render_mode = ENV_wrapper.HUMAN_RENDER)
        else:
            self.env = gym.make(self.env_name)
        
        if self.is_norm_wrapper:
            self.env = gym.wrappers.NormalizeObservation(self.env)
        
        self.act_diff = act_max - act_min
        self.act_min = act_min

    def _preprocess_action(self, action) -> float:
        
        if self.algorithm == 'PPO' or self.algorithm == 'RARL_PPO':
            return np.array(action.squeeze(0).cpu()) * self.act_diff + self.act_min # scale action in range [act_min, act_max]
        if self.algorithm == 'SAC' or self.algorithm == 'RARL_SAC':
            return action.detach().cpu().numpy()[0]

    def _postprocess_state(self, state) -> torch.tensor:
        return torch.tensor(state, dtype=torch.float32).reshape(1,-1)

    def step(self, action):
        state, reward, terminated, truncated, info = self.env.step(self._preprocess_action(action))

        return self._postprocess_state(state), reward, terminated, truncated, info

    def reset(self):
        state, info = self.env.reset()
        return self._postprocess_state(state), info


class ENV_Adversarial_wrapper(gym.Wrapper):

    HUMAN_RENDER = 'human'
    
    def __init__(self, env_name : str,
                   action_range : list[float, float], 
               adv_action_range : list[float, float], 
                    render_mode : str    = None, 
                is_norm_wrapper : bool   = True, 
                algorithm : str = 'SAC') -> None:
               

        self.env_name = env_name
        self.is_norm_wrapper = is_norm_wrapper
        self.algorithm = algorithm

        if render_mode == ENV_wrapper.HUMAN_RENDER:
            self.env = gym.make(self.env_name, render_mode = ENV_wrapper.HUMAN_RENDER)
        else:
            self.env = gym.make(self.env_name)
        
        if self.is_norm_wrapper:
            self.env = gym.wrappers.NormalizeObservation(self.env)
        
        self.action_range = action_range
        self.adv_action_range = adv_action_range

    def _scale_action(self, action, adv_action_range):
        
         if self.algorithm == 'PPO' or self.algorithm == 'RARL_PPO' or self.algorithm == 'RARL':
            return (action * (adv_action_range[1] - adv_action_range[0])) + adv_action_range[0] # scale action in range [act_min, act_max]
        
         if self.algorithm == 'SAC' or self.algorithm == 'RARL_SAC':
             return action
        
    def _preprocess_action(self, action) -> float:
        return np.array(action.squeeze(0).cpu().numpy()) # scale action in range [act_min, act_max]

    def _postprocess_state(self, state) -> torch.tensor:
        return torch.tensor(state, dtype=torch.float32).reshape(1,-1)

    def step(self, action, action_adv = None):
        if action_adv == None:
            action_for_env = self._scale_action(action, self.action_range)
        else:
            action_for_env = self._scale_action(action, self.action_range) + self._scale_action(action_adv, self.adv_action_range)
            
        state, reward, terminated, truncated, info = self.env.step(self._preprocess_action(action_for_env))

        return self._postprocess_state(state), reward, terminated, truncated, info

    def reset(self):
        state, info = self.env.reset()
        return self._postprocess_state(state), info
    
    
    
class Walker_env_pert(ENV_Adversarial_wrapper):
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
