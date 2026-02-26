import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

# —— 让 Normal 在多维动作上返回 summed log_prob 与 entropy（保持和原工程一致）——
MultiVariateNormal = torch.distributions.Normal
_temp_logprob = MultiVariateNormal.log_prob
MultiVariateNormal.log_prob = lambda self, val: _temp_logprob(self, val).sum(-1, keepdim=True)
_temp_entropy = MultiVariateNormal.entropy
MultiVariateNormal.entropy = lambda self: _temp_entropy(self).sum(-1)
MultiVariateNormal.mode = lambda self: self.mean

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
Tensor = FloatTensor

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.zero_()

# =========================
# 1) MuscleNN
# =========================
class MuscleNN(nn.Module):
    def __init__(self,num_total_muscle_related_dofs,num_dofs,num_muscles):
        super(MuscleNN,self).__init__()
        self.num_total_muscle_related_dofs = num_total_muscle_related_dofs
        self.num_dofs = num_dofs
        self.num_muscles = num_muscles

        num_h1 = 1024
        num_h2 = 512
        num_h3 = 512
        self.fc = nn.Sequential(
            nn.Linear(num_total_muscle_related_dofs+num_dofs,num_h1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(num_h1,num_h2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(num_h2,num_h3),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(num_h3,num_muscles),
            nn.Tanh(),
            nn.ReLU()
        )
        self.std_muscle_tau = torch.zeros(self.num_total_muscle_related_dofs)
        self.std_tau = torch.zeros(self.num_dofs)
        for i in range(self.num_total_muscle_related_dofs):
            self.std_muscle_tau[i] = 200.0
        for i in range(self.num_dofs):
            self.std_tau[i] = 200.0

        if use_cuda:
            self.std_tau = self.std_tau.cuda()
            self.std_muscle_tau = self.std_muscle_tau.cuda()
            self.cuda()
        self.fc.apply(weights_init)

    def forward(self,muscle_tau,tau):
        muscle_tau = muscle_tau/self.std_muscle_tau
        tau = tau/self.std_tau
        out = self.fc.forward(torch.cat([muscle_tau,tau],dim=1))
        return out

    def load(self,path):
        print('load muscle nn {}'.format(path))
        self.load_state_dict(torch.load(path))

    def save(self,path):
        print('save muscle nn {}'.format(path))
        torch.save(self.state_dict(),path)

    def get_activation(self,muscle_tau,tau):
        act = self.forward(Tensor(muscle_tau.reshape(1,-1).astype(np.float32)),
                           Tensor(tau.reshape(1,-1).astype(np.float32)))
        return act.cpu().detach().numpy().squeeze()


# ===========================================
# 2) SimulationNN: PD policy + value function
# ===========================================
class SimulationNN(nn.Module):
    def __init__(self,num_states,num_actions):
        super(SimulationNN,self).__init__()
        self.num_actions = num_actions

        h1, h2 = 256, 256
        # policy (PD)
        self.p_fc1 = nn.Linear(num_states, h1)
        self.p_fc2 = nn.Linear(h1, h2)
        self.p_mean = nn.Linear(h2, num_actions)
        self.log_std = nn.Parameter(torch.zeros(num_actions))

        # value
        self.v_fc1 = nn.Linear(num_states, h1)
        self.v_fc2 = nn.Linear(h1, h2)
        self.v_out = nn.Linear(h2, 1)

        # init
        for m in [self.p_fc1,self.p_fc2,self.p_mean,self.v_fc1,self.v_fc2,self.v_out]:
            weights_init(m)

    def forward(self,x):
        # PD policy
        p = F.relu(self.p_fc1(x))
        p = F.relu(self.p_fc2(p))
        dist_pd = MultiVariateNormal(self.p_mean(p), self.log_std.exp())

        # value
        v = F.relu(self.v_fc1(x))
        v = F.relu(self.v_fc2(v))
        v = self.v_out(v)
        return dist_pd, v

    def load(self,path):
        print('load simulation nn {}'.format(path))
        self.load_state_dict(torch.load(path))

    def save(self,path):
        print('save simulation nn {}'.format(path))
        torch.save(self.state_dict(),path)

    def get_action(self,s):
        ts = torch.tensor(s.astype(np.float32))
        p,_ = self.forward(ts)
        return p.loc.cpu().detach().numpy().squeeze()

    def get_random_action(self,s):
        ts = torch.tensor(s.astype(np.float32))
        p,_ = self.forward(ts)
        return p.sample().cpu().detach().numpy().squeeze()


# ==================================
# 3) ExoPolicyNN: exoskeleton policy
# ==================================
class ExoPolicyNN(nn.Module):
    def __init__(self, num_states, num_actions):
        super(ExoPolicyNN, self).__init__()
        self.num_actions = num_actions
        h1, h2 = 256, 256
        self.fc1 = nn.Linear(num_states, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.mean = nn.Linear(h2, num_actions)
        self.log_std = nn.Parameter(torch.zeros(num_actions))

        for m in [self.fc1,self.fc2,self.mean]:
            weights_init(m)

    def forward(self, x):
        z = F.relu(self.fc1(x))
        z = F.relu(self.fc2(z))
        dist_exo = MultiVariateNormal(self.mean(z), self.log_std.exp())
        return dist_exo

    def load(self,path):
        print('load exo nn {}'.format(path))
        self.load_state_dict(torch.load(path))

    def save(self,path):
        print('save exo nn {}'.format(path))
        torch.save(self.state_dict(),path)
