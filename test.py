import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box,Dict
from gymnasium.wrappers import Autoreset
import gymnasium_robotics
gym.register_envs(gymnasium_robotics)

class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.LazyLinear(256)
        self.l2 = nn.LazyLinear(256)
        self.l3 = nn.LazyLinear(256)
        self.output = nn.LazyLinear(3)
    
    def forward(self,obs: Tensor):
        obs = F.relu(self.l1(obs))
        obs = F.relu(self.l2(obs))
        obs = F.relu(self.l3(obs))
        output = F.tanh(self.output(obs))
        return output
    
model = Actor()
model.forward(torch.rand((1,6),dtype=torch.float32).to("cpu"))
chk = torch.load(".\data\model_2.pth")
model.load_state_dict(chk.get("actor state"))

class FetchReachCustom(gym.Wrapper):
    def __init__(self,env : gym.Env):
        super().__init__(env)
        self.action_space = Box(-1,1,(3,),np.float32)
        self.observation_space = Dict(
            {
            "observation" : Box(-np.inf,np.inf,(3,),np.float64),
            "achieved_goal" : Box(-np.inf,np.inf,(3,),np.float64),
            "desired_goal" : Box(-np.inf,np.inf,(3,),np.float64)
            }
        )
    
    def process_obs(self,observation):
        observation["observation"] = observation["observation"][:3]
        return observation
         
    def step(self, action):
        action = np.append(action,0)
        observation, reward, done,truncated, info = self.env.step(action)
        return  self.process_obs(observation), reward, done,truncated, info
    
    def reset(self,seed=None, options=None):
        observation,info = self.env.reset(seed=seed,options=options)
        self.env.unwrapped.data.qpos[0] = 0.3
        self.env.unwrapped.data.qpos[1] = 0.5
        return self.process_obs(observation),info

def tranform_observation(observation_dict : Dict): 
    #observation = observation_dict.get("observation")
    current_pos = observation_dict.get("achieved_goal")
    target = observation_dict.get("desired_goal")
    output = np.concatenate((current_pos,target),axis=-1)
    return torch.from_numpy(output).to(device="cpu",dtype=torch.float32)

env = gym.make("FetchReachDense-v3",render_mode="human",max_episode_steps=100)
env = FetchReachCustom(env)
env = Autoreset(env)

obs,_ = env.reset()
previous_action = 0
for _ in range(10000):
    obss = tranform_observation(obs)
    action = model(obss).detach().numpy() 
    #smth = 0.99*previous_action + (1 - 0.99)*action
    obs,_,done,trunc,_ = env.step(action)
    previous_action = action
    env.render()
env.close()