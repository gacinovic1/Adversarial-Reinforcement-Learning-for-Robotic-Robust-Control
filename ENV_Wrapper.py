import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import mujoco
import numpy as np


class ENV_wrapper(gym.Wrapper):

    HUMAN_RENDER = 'human'
    
    def __init__(self, env_name : str,
                        act_min : float, 
                        act_max : float, 
                    render_mode : str    = None, 
                is_norm_wrapper : bool   = True, algorithm : str = 'PPO'
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
            return action.detach().cpu().numpy()[0]* self.act_diff + self.act_min

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
                algorithm : str = 'RARL_PPO') -> None:
               

        self.env_name = env_name
        self.is_norm_wrapper = is_norm_wrapper
        self.algorithm = algorithm
        self.rendering = render_mode == ENV_Adversarial_wrapper.HUMAN_RENDER
        self.ids = {}

        if self.rendering:
            self.env = gym.make(self.env_name, render_mode = ENV_Adversarial_wrapper.HUMAN_RENDER) #,visual_options = {mujoco.mjtVisFlag.mjVIS_CONTACTPOINT: True, mujoco.mjtVisFlag.mjVIS_CONTACTFORCE: True})
        else:
            self.env = gym.make(self.env_name)
        
        if self.is_norm_wrapper:
            self.env = gym.wrappers.NormalizeObservation(self.env)
        
        self.action_range = action_range
        self.adv_action_range = adv_action_range
        self.mj_model, self.mj_data = self.env.unwrapped.model, self.env.unwrapped.data
        self.last_force = None
        self.last_force_body = None

    def _scale_action(self, action, adv_action_range):
        
         if self.algorithm == 'PPO' or self.algorithm == 'RARL_PPO' or self.algorithm == 'RARL':
            return (action * (adv_action_range[1] - adv_action_range[0])) + adv_action_range[0] # scale action in range [act_min, act_max]
        
         if self.algorithm == 'SAC' or self.algorithm == 'RARL_SAC':
             return (action * (adv_action_range[1] - adv_action_range[0])) + adv_action_range[0] 
        
    def _preprocess_action(self, action) -> float:
        return np.array(action.squeeze(0).cpu().detach().numpy()) # scale action in range [act_min, act_max]

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
    
    def render_forces(self):

        if not self.rendering: return
        if self.last_force is None:return
        scene = self.env.unwrapped.mujoco_renderer.viewer.scn

        if scene.ngeom >= scene.maxgeom:
            return

        body_id = self.last_force_body
        pos = self.mj_data.xpos[body_id].copy()

        force = self.last_force[:3]
        norm = np.linalg.norm(force)
        if norm < 1e-6:
            print("Force too small to be rendered")
            return

        direction = force / norm
        scale = 1.0
        end = pos + scale * direction

        rgba = np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float32)
        radius = 0.08
        scene.ngeom += 1
        geom = scene.geoms[scene.ngeom - 1]
        mujoco.mjv_initGeom(geom,mujoco.mjtGeom.mjGEOM_ARROW,np.zeros(3, dtype=np.float64),np.zeros(3, dtype=np.float64),np.zeros(9, dtype=np.float64),rgba)
        mujoco.mjv_connector(geom,mujoco.mjtGeom.mjGEOM_ARROW,radius,pos.astype(np.float64),end.astype(np.float64))
    
    def render(self):
        
        out = self.env.unwrapped.render()
        self.render_forces()
        return out
    
    
    