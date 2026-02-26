# -*- coding: utf-8 -*-
"""
Unified MASS training script.
  --mode original   (default)  Standard PD + Muscle training
  --mode exo                   PD + Exoskeleton + Muscle training
"""
import math
import random
import time
import csv
import os
import sys
from datetime import datetime
from collections import namedtuple, deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

import pymss  # type: ignore
from Model import SimulationNN, MuscleNN, ExoPolicyNN, use_cuda, FloatTensor, Tensor

LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor  = torch.cuda.ByteTensor  if use_cuda else torch.ByteTensor

Episode     = namedtuple('Episode',('s','a','r','value','logprob'))
Transition  = namedtuple('Transition',('s','a','logprob','TD','GAE'))

# ---------- helpers ----------
def _ensure_dir(p):
    d = os.path.dirname(p)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

class EpisodeBuffer(object):
    def __init__(self):
        self.data = []
    def Push(self, *args):
        self.data.append(Episode(*args))
    def Pop(self):
        if self.data:
            self.data.pop()
    def GetData(self):
        return self.data
    def Clear(self):
        self.data.clear()

class ReplayBuffer(object):
    def __init__(self, buff_size=10000):
        self.buffer = deque(maxlen=buff_size)
    def Push(self,*args):
        self.buffer.append(Transition(*args))
    def Clear(self):
        self.buffer.clear()


class PPO(object):
    def __init__(self, meta_file, mode='original'):
        np.random.seed(seed=int(time.time()))
        self.mode = mode
        self.num_slaves = 16
        self.env = pymss.pymss(meta_file, self.num_slaves)

        self.use_muscle   = self.env.UseMuscle()
        self.num_state    = self.env.GetNumState()
        self.num_action   = self.env.GetNumAction()
        self.num_muscles  = self.env.GetNumMuscles()

        # Exo action space (only used in exo mode)
        if self.mode == 'exo':
            self.env.SetUseExo(True)
            try:
                exo_mat = self.env.GetExoTorques()
                self.num_exo_action = exo_mat.shape[1]
            except:
                self.num_exo_action = 6
        else:
            self.env.SetUseExo(False)
            self.num_exo_action = 6  # placeholder, not used

        self.num_epochs          = 10
        self.num_epochs_muscle   = 3
        self.num_evaluation      = 0
        self.num_tuple_so_far    = 0
        self.num_episode         = 0
        self.num_tuple           = 0
        self.num_simulation_Hz   = self.env.GetSimulationHz()
        self.num_control_Hz      = self.env.GetControlHz()
        self.num_sim_per_control = self.num_simulation_Hz // self.num_control_Hz

        self.gamma = 0.99
        self.lb    = 0.99

        self.buffer_size       = 2048
        self.batch_size        = 128
        self.muscle_batch_size = 128
        self.replay_buffer     = ReplayBuffer(30000)
        self.muscle_buffer     = {}

        # Networks
        self.model = SimulationNN(self.num_state, self.num_action)
        self.muscle_model = MuscleNN(self.env.GetNumTotalMuscleRelatedDofs(),
                                     self.num_action, self.num_muscles)

        if self.mode == 'exo':
            self.exo_model = ExoPolicyNN(self.num_state, self.num_exo_action)
            if use_cuda:
                self.model.cuda(); self.exo_model.cuda(); self.muscle_model.cuda()
            # Joint optimizer for PD + Exo
            self.learning_rate = 1e-4
            self.clip_ratio    = 0.2
            self.optimizer = optim.Adam(
                list(self.model.parameters()) + list(self.exo_model.parameters()),
                lr=self.learning_rate)
        else:
            self.exo_model = None
            if use_cuda:
                self.model.cuda(); self.muscle_model.cuda()
            self.learning_rate = 1e-4
            self.clip_ratio    = 0.2
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        self.optimizer_muscle = optim.Adam(self.muscle_model.parameters(), lr=self.learning_rate)
        self.max_iteration    = 50000
        self.w_entropy        = -0.001

        # Stats
        self.loss_actor  = 0.0
        self.loss_critic = 0.0
        self.loss_muscle = 0.0
        self.rewards     = []
        self.sum_return  = 0.0
        self.max_return  = -1.0
        self.max_return_epoch = 1
        self.tic = time.time()

        self.episodes = [EpisodeBuffer() for _ in range(self.num_slaves)]

        # Per-episode stat accumulators (control-step level)
        def _new_ep_stat():
            return {
                'tau_abs_sum': 0.0,
                'tau_sq_sum':  0.0,
                'tau_abs_max': 0.0,
                'act_abs_sum': 0.0,
                'act_sq_sum':  0.0,
                'act_abs_max': 0.0,
                'exo_abs_sum': 0.0,
                'exo_sq_sum':  0.0,
                'exo_abs_max': 0.0,
                'step_count':  0,
                'ret_sum':     0.0
            }
        self.ep_stats = [_new_ep_stat() for _ in range(self.num_slaves)]

        # CSV init (baseline always; exo CSV only in exo mode)
        self.csv_baseline = os.path.join('..','nn','episode_metrics_baseline.csv')
        _ensure_dir(self.csv_baseline)
        if not os.path.exists(self.csv_baseline):
            with open(self.csv_baseline,'w',newline='') as f:
                w = csv.writer(f)
                w.writerow([
                    'eval_round','slave_id',
                    'avg_tau_abs','max_tau_abs','sum_tau_sq',
                    'avg_activation','max_activation','sum_activation_sq',
                    'episode_return','steps_in_episode',
                    'ctrl_avg_power','ctrl_energy'
                ])

        if self.mode == 'exo':
            self.csv_exo = os.path.join('..','nn','episode_metrics.csv')
            _ensure_dir(self.csv_exo)
            if not os.path.exists(self.csv_exo):
                with open(self.csv_exo,'w',newline='') as f:
                    w = csv.writer(f)
                    w.writerow([
                        'eval_round','slave_id',
                        'avg_tau_abs','max_tau_abs','sum_tau_sq',
                        'avg_activation','max_activation','sum_activation_sq',
                        'episode_return','steps_in_episode',
                        'noise_pd','noise_exo',
                        'exo_tau_avg_abs','exo_tau_max_abs','exo_tau_sum_sq',
                        'exo_avg_power','exo_energy',
                        'ctrl_avg_power','ctrl_energy'
                    ])

        self.env.Resets(True)

    # ---------- Save / Load ----------
    def SaveModel(self):
        self.model.save('../nn/current.pt')
        self.muscle_model.save('../nn/current_muscle.pt')
        if self.mode == 'exo':
            self.exo_model.save('../nn/current_exo.pt')

        if self.max_return_epoch == self.num_evaluation:
            self.model.save('../nn/max.pt')
            self.muscle_model.save('../nn/max_muscle.pt')
            if self.mode == 'exo':
                self.exo_model.save('../nn/max_exo.pt')

        if self.num_evaluation % 100 == 0:
            k = self.num_evaluation // 100
            self.model.save('../nn/{}.pt'.format(k))
            self.muscle_model.save('../nn/{}_muscle.pt'.format(k))
            if self.mode == 'exo':
                self.exo_model.save('../nn/{}_exo.pt'.format(k))

    def LoadModel(self, path):
        self.model.load('../nn/{}.pt'.format(path))
        self.muscle_model.load('../nn/{}_muscle.pt'.format(path))
        if self.mode == 'exo':
            self.exo_model.load('../nn/{}_exo.pt'.format(path))

    # ---------- GAE ----------
    def ComputeTDandGAE(self):
        self.replay_buffer.Clear()
        self.muscle_buffer = {}
        self.sum_return = 0.0

        for epi in self.total_episodes:
            data = epi.GetData()
            n = len(data)
            if n == 0:
                continue
            states, actions, rewards, values, logprobs = zip(*data)
            values = np.concatenate((values, np.zeros(1)), axis=0)
            adv = np.zeros(n); ad_t = 0.0

            epi_return = 0.0
            for i in reversed(range(n)):
                epi_return += rewards[i]
                delta = rewards[i] + values[i+1]*self.gamma - values[i]
                ad_t  = delta + self.gamma*self.lb*ad_t
                adv[i]= ad_t
            self.sum_return += epi_return
            TD = values[:n] + adv
            for i in range(n):
                self.replay_buffer.Push(states[i], actions[i], logprobs[i], TD[i], adv[i])

        self.num_episode = len(self.total_episodes)
        self.num_tuple   = len(self.replay_buffer.buffer)
        print('SIM : {}'.format(self.num_tuple))
        self.num_tuple_so_far += self.num_tuple

        self.env.ComputeMuscleTuples()
        self.muscle_buffer['JtA']    = self.env.GetMuscleTuplesJtA()
        self.muscle_buffer['TauDes'] = self.env.GetMuscleTuplesTauDes()
        self.muscle_buffer['L']      = self.env.GetMuscleTuplesL()
        self.muscle_buffer['b']      = self.env.GetMuscleTuplesb()

    # ---------- Generate Transitions ----------
    def GenerateTransitions(self):
        self.total_episodes = []
        states = self.env.GetStates()
        local_step = 0
        counter = 0

        ep_rows_baseline = []
        ep_rows_exo      = []

        def _flush_episode_row(slave_id, eval_round, noise_pd, noise_exo):
            st = self.ep_stats[slave_id]
            steps = max(1, st['step_count'])

            avg_tau_abs = st['tau_abs_sum'] / steps
            max_tau_abs = st['tau_abs_max']
            sum_tau_sq  = st['tau_sq_sum']
            avg_act_abs = st['act_abs_sum'] / steps if self.use_muscle else float('nan')
            max_act_abs = st['act_abs_max']          if self.use_muscle else float('nan')
            sum_act_sq  = st['act_sq_sum']           if self.use_muscle else 0.0

            try:
                ctrl_avg_power = float(self.env.GetEpisodeCtrlAvgPowerVec()[slave_id])
                ctrl_energy    = float(self.env.GetEpisodeCtrlEnergyVec()[slave_id])
            except:
                ctrl_avg_power = float('nan')
                ctrl_energy    = float('nan')

            row_base = [
                eval_round, slave_id,
                avg_tau_abs, max_tau_abs, sum_tau_sq,
                avg_act_abs, max_act_abs, sum_act_sq,
                st['ret_sum'], st['step_count'],
                ctrl_avg_power, ctrl_energy
            ]
            ep_rows_baseline.append(row_base)

            if self.mode == 'exo':
                exo_avg_abs = st['exo_abs_sum'] / steps
                exo_max_abs = st['exo_abs_max']
                exo_sum_sq  = st['exo_sq_sum']
                try:
                    exo_avg_power = float(self.env.GetEpisodeExoAvgPowerVec()[slave_id])
                    exo_energy    = float(self.env.GetEpisodeExoEnergyVec()[slave_id])
                except:
                    exo_avg_power = float('nan')
                    exo_energy    = float('nan')

                row_exo = [
                    eval_round, slave_id,
                    avg_tau_abs, max_tau_abs, sum_tau_sq,
                    avg_act_abs, max_act_abs, sum_act_sq,
                    st['ret_sum'], st['step_count'],
                    noise_pd, noise_exo,
                    exo_avg_abs, exo_max_abs, exo_sum_sq,
                    exo_avg_power, exo_energy,
                    ctrl_avg_power, ctrl_energy
                ]
                ep_rows_exo.append(row_exo)

            # Reset stats for this slave
            self.ep_stats[slave_id] = {
                k: (0.0 if isinstance(v, float) else 0)
                for k, v in self.ep_stats[slave_id].items()
            }

        while True:
            counter += 1
            if counter % 10 == 0:
                print('SIM : {}'.format(local_step), end='\r')

            if self.mode == 'exo':
                # Two policy heads: PD + Exo
                a_dist_pd, v = self.model(Tensor(states))
                a_dist_exo   = self.exo_model(Tensor(states))

                a_pd  = a_dist_pd.sample()
                a_exo = a_dist_exo.sample()

                actions_combined = torch.cat([a_pd, a_exo], dim=1).cpu().detach().numpy()
                logprob_pd  = a_dist_pd.log_prob(a_pd).cpu().detach().numpy().reshape(-1)
                logprob_exo = a_dist_exo.log_prob(a_exo).cpu().detach().numpy().reshape(-1)
                logprobs    = logprob_pd + logprob_exo
                values      = v.cpu().detach().numpy().reshape(-1)

                self.env.SetActions(a_pd.cpu().detach().numpy())
                self.env.SetExoTorques(a_exo.cpu().detach().numpy())
            else:
                # Original: single PD head
                a_dist, v = self.model(Tensor(states))
                actions = a_dist.sample().cpu().detach().numpy()
                logprobs = a_dist.log_prob(Tensor(actions)).cpu().detach().numpy().reshape(-1)
                values   = v.cpu().detach().numpy().reshape(-1)
                self.env.SetActions(actions)

            # Torque statistics (control-step)
            dt_np = self.env.GetDesiredTorques()
            tau_abs = np.abs(dt_np)
            tau_sq  = dt_np * dt_np
            mean_abs_tau = tau_abs.mean(axis=1)
            max_abs_tau  = tau_abs.max(axis=1)
            sum_tau_sq   = tau_sq.sum(axis=1)

            if self.mode == 'exo':
                exo_np  = a_exo.cpu().detach().numpy()
                exo_abs = np.abs(exo_np)
                exo_sq  = exo_np * exo_np

            for j in range(self.num_slaves):
                st = self.ep_stats[j]
                st['tau_abs_sum'] += float(mean_abs_tau[j])
                st['tau_sq_sum']  += float(sum_tau_sq[j])
                st['tau_abs_max']  = max(st['tau_abs_max'], float(max_abs_tau[j]))
                if self.mode == 'exo':
                    st['exo_abs_sum'] += float(exo_abs[j].mean())
                    st['exo_sq_sum']  += float(exo_sq[j].sum())
                    st['exo_abs_max']  = max(st['exo_abs_max'], float(exo_abs[j].max()))

            # Muscle sub-steps
            if self.use_muscle:
                mt = Tensor(self.env.GetMuscleTorques())
                for _ in range(self.num_sim_per_control // 2):
                    dt = Tensor(self.env.GetDesiredTorques())
                    activations = self.muscle_model(mt, dt).cpu().detach().numpy()
                    self.env.SetActivationLevels(activations)

                    act_abs = np.abs(activations)
                    act_sq  = activations * activations
                    mean_abs_act = act_abs.mean(axis=1)
                    max_abs_act  = act_abs.max(axis=1)
                    sum_act_sq   = act_sq.sum(axis=1)

                    dt_np_sub = dt.cpu().detach().numpy()
                    tau_abs_sub_max = np.abs(dt_np_sub).max(axis=1)

                    for j in range(self.num_slaves):
                        st = self.ep_stats[j]
                        st['act_abs_sum'] += float(mean_abs_act[j])
                        st['act_sq_sum']  += float(sum_act_sq[j])
                        st['act_abs_max']  = max(st['act_abs_max'], float(max_abs_act[j]))
                        st['tau_abs_max']  = max(st['tau_abs_max'], float(tau_abs_sub_max[j]))

                    self.env.Steps(2)
            else:
                self.env.StepsAtOnce()

            # Noise for CSV
            try:    noise_pd = self.model.log_std.exp().mean().item()
            except: noise_pd = float('nan')
            if self.mode == 'exo':
                try:    noise_exo = self.exo_model.log_std.exp().mean().item()
                except: noise_exo = float('nan')
            else:
                noise_exo = float('nan')

            # Reward and termination
            for j in range(self.num_slaves):
                nan_occur = False
                terminated_state = True

                if self.mode == 'exo':
                    a_check = actions_combined[j]
                else:
                    a_check = actions[j]

                if (np.any(np.isnan(states[j])) or np.any(np.isnan(a_check)) or
                    np.any(np.isnan(values[j])) or np.any(np.isnan(logprobs[j]))):
                    nan_occur = True
                elif self.env.IsEndOfEpisode(j) is False:
                    terminated_state = False
                    r = self.env.GetReward(j)

                    if self.mode == 'exo':
                        self.episodes[j].Push(states[j], actions_combined[j], r, values[j], logprobs[j])
                    else:
                        self.episodes[j].Push(states[j], a_check, r, values[j], logprobs[j])

                    st = self.ep_stats[j]
                    st['ret_sum']    += float(r)
                    st['step_count'] += 1
                    local_step += 1

                if terminated_state or nan_occur:
                    if nan_occur:
                        self.episodes[j].Pop()
                    _flush_episode_row(j, self.num_evaluation + 1, noise_pd, noise_exo)

                    self.total_episodes.append(self.episodes[j])
                    self.episodes[j] = EpisodeBuffer()
                    self.env.Reset(True, j)

            if local_step >= self.buffer_size:
                break

            states = self.env.GetStates()

        # Flush CSV rows
        if ep_rows_baseline:
            with open(self.csv_baseline, 'a', newline='') as f:
                csv.writer(f).writerows(ep_rows_baseline)
        if self.mode == 'exo' and ep_rows_exo:
            with open(self.csv_exo, 'a', newline='') as f:
                csv.writer(f).writerows(ep_rows_exo)

    # ---------- Optimize SimulationNN (+ ExoPolicyNN if exo mode) ----------
    def OptimizeSimulationNN(self):
        all_trans = np.array(self.replay_buffer.buffer, dtype=object)
        for e in range(self.num_epochs):
            np.random.shuffle(all_trans)
            for i in range(len(all_trans) // self.batch_size):
                transitions = all_trans[i*self.batch_size:(i+1)*self.batch_size]
                batch = Transition(*zip(*transitions))
                stack_s   = np.vstack(batch.s).astype(np.float32)
                stack_a   = np.vstack(batch.a).astype(np.float32)
                stack_lp  = np.vstack(batch.logprob).astype(np.float32)
                stack_td  = np.vstack(batch.TD).astype(np.float32)
                stack_gae = np.vstack(batch.GAE).astype(np.float32)

                if self.mode == 'exo':
                    a_pd  = Tensor(stack_a[:, :self.num_action])
                    a_exo = Tensor(stack_a[:, self.num_action:])

                    a_dist_pd, v = self.model(Tensor(stack_s))
                    a_dist_exo   = self.exo_model(Tensor(stack_s))
                    logprob_now  = a_dist_pd.log_prob(a_pd) + a_dist_exo.log_prob(a_exo)
                    ratio        = torch.exp(logprob_now - Tensor(stack_lp))

                    stack_gae = (stack_gae - stack_gae.mean()) / (stack_gae.std() + 1e-5)
                    stack_gae = Tensor(stack_gae)
                    stack_td  = Tensor(stack_td)

                    loss_critic  = ((v - stack_td).pow(2)).mean()
                    surrogate1   = ratio * stack_gae
                    surrogate2   = torch.clamp(ratio, 1.0-self.clip_ratio, 1.0+self.clip_ratio) * stack_gae
                    loss_actor   = -torch.min(surrogate1, surrogate2).mean()
                    loss_entropy = -self.w_entropy * (a_dist_pd.entropy().mean() + a_dist_exo.entropy().mean())

                    self.loss_actor  = loss_actor.cpu().detach().numpy().tolist()
                    self.loss_critic = loss_critic.cpu().detach().numpy().tolist()

                    loss = loss_actor + loss_entropy + loss_critic
                    self.optimizer.zero_grad()
                    loss.backward(retain_graph=True)
                    for p in list(self.model.parameters()) + list(self.exo_model.parameters()):
                        if p.grad is not None:
                            p.grad.data.clamp_(-0.5, 0.5)
                    self.optimizer.step()
                else:
                    a_dist, v = self.model(Tensor(stack_s))
                    loss_critic = ((v - Tensor(stack_td)).pow(2)).mean()

                    ratio = torch.exp(a_dist.log_prob(Tensor(stack_a)) - Tensor(stack_lp))
                    stack_gae = (stack_gae - stack_gae.mean()) / (stack_gae.std() + 1e-5)
                    stack_gae = Tensor(stack_gae)
                    surrogate1 = ratio * stack_gae
                    surrogate2 = torch.clamp(ratio, min=1.0-self.clip_ratio, max=1.0+self.clip_ratio) * stack_gae
                    loss_actor = -torch.min(surrogate1, surrogate2).mean()
                    loss_entropy = -self.w_entropy * a_dist.entropy().mean()

                    self.loss_actor  = loss_actor.cpu().detach().numpy().tolist()
                    self.loss_critic = loss_critic.cpu().detach().numpy().tolist()

                    loss = loss_actor + loss_entropy + loss_critic
                    self.optimizer.zero_grad()
                    loss.backward(retain_graph=True)
                    for param in self.model.parameters():
                        if param.grad is not None:
                            param.grad.data.clamp_(-0.5, 0.5)
                    self.optimizer.step()

            if self.mode == 'exo':
                print('Optimizing sim/exo nn : {}/{}'.format(e+1, self.num_epochs), end='\r')
            else:
                print('Optimizing sim nn : {}/{}'.format(e+1, self.num_epochs), end='\r')
        print('')

    def _generate_shuffle_indices(self, n, m):
        p = np.random.permutation(n)
        r = m - n % m
        if r > 0:
            p = np.hstack([p, np.random.randint(0, n, r)])
        return p.reshape(-1, m)

    def OptimizeMuscleNN(self):
        for e in range(self.num_epochs_muscle):
            mb = self._generate_shuffle_indices(self.muscle_buffer['JtA'].shape[0], self.muscle_batch_size)
            for idx in mb:
                JtA     = Tensor(self.muscle_buffer['JtA'][idx].astype(np.float32))
                tau_des = Tensor(self.muscle_buffer['TauDes'][idx].astype(np.float32))
                L       = Tensor(self.muscle_buffer['L'][idx].astype(np.float32))
                L       = L.reshape(self.muscle_batch_size, self.num_action, self.num_muscles)
                b       = Tensor(self.muscle_buffer['b'][idx].astype(np.float32))

                activation = self.muscle_model(JtA, tau_des)
                tau = torch.einsum('ijk,ik->ij', (L, activation)) + b

                loss_reg    = (activation).pow(2).mean()
                loss_target = (((tau - tau_des) / 100.0).pow(2)).mean()
                loss = 0.01 * loss_reg + loss_target

                self.optimizer_muscle.zero_grad()
                loss.backward(retain_graph=True)
                for p in self.muscle_model.parameters():
                    if p.grad is not None:
                        p.grad.data.clamp_(-0.5, 0.5)
                self.optimizer_muscle.step()
            print('Optimizing muscle nn : {}/{}'.format(e+1, self.num_epochs_muscle), end='\r')
        self.loss_muscle = loss.cpu().detach().numpy().tolist()
        print('')

    def OptimizeModel(self):
        self.ComputeTDandGAE()
        self.OptimizeSimulationNN()
        if self.use_muscle:
            self.OptimizeMuscleNN()

    def Train(self):
        self.GenerateTransitions()
        self.OptimizeModel()

    def Evaluate(self):
        self.num_evaluation += 1
        elapsed = time.time() - self.tic
        h = int(elapsed // 3600.0)
        m = int(elapsed // 60.0) - h * 60
        s = int(elapsed) - h * 3600 - m * 60
        if self.num_episode == 0: self.num_episode = 1
        if self.num_tuple   == 0: self.num_tuple   = 1
        avg_return = self.sum_return / self.num_episode
        if self.max_return < avg_return:
            self.max_return = avg_return
            self.max_return_epoch = self.num_evaluation

        try:    noise_pd = self.model.log_std.exp().mean().item()
        except: noise_pd = float('nan')
        if self.mode == 'exo':
            try:    noise_exo = self.exo_model.log_std.exp().mean().item()
            except: noise_exo = float('nan')

        print('# {} === {}h:{}m:{}s === [mode={}]'.format(self.num_evaluation, h, m, s, self.mode))
        print('||Loss Actor               : {:.4f}'.format(self.loss_actor))
        print('||Loss Critic              : {:.4f}'.format(self.loss_critic))
        print('||Loss Muscle              : {:.4f}'.format(self.loss_muscle))
        print('||Noise (PD)               : {:.3f}'.format(noise_pd))
        if self.mode == 'exo':
            print('||Noise (Exo)              : {:.3f}'.format(noise_exo))
        print('||Num Transition So far    : {}'.format(self.num_tuple_so_far))
        print('||Num Transition           : {}'.format(self.num_tuple))
        print('||Num Episode              : {}'.format(self.num_episode))
        print('||Avg Return per episode   : {:.3f}'.format(avg_return))
        print('||Avg Reward per transition: {:.3f}'.format(self.sum_return / self.num_tuple))
        print('||Avg Step per episode     : {:.1f}'.format(self.num_tuple / self.num_episode))
        print('||Max Avg Return So far    : {:.3f} at #{}'.format(self.max_return, self.max_return_epoch))
        self.rewards.append(avg_return)

        self.SaveModel()
        print('=============================================')
        return np.array(self.rewards)


# ===== Plotting =====
import matplotlib
import matplotlib.pyplot as plt
plt.ion()

def Plot(y, title, num_fig=1, ylim=True):
    temp_y = np.zeros(y.shape)
    if y.shape[0] > 5:
        temp_y[0] = y[0]
        temp_y[1] = 0.5 * (y[0] + y[1])
        temp_y[2] = 0.3333 * (y[0] + y[1] + y[2])
        temp_y[3] = 0.25 * (y[0] + y[1] + y[2] + y[3])
        for i in range(4, y.shape[0]):
            temp_y[i] = np.sum(y[i-4:i+1]) * 0.2
    plt.figure(num_fig); plt.clf(); plt.title(title)
    plt.plot(y, 'b'); plt.plot(temp_y, 'r')
    if ylim: plt.ylim([0, 1])
    plt.show(); plt.pause(0.001)
    if y.shape[0] % 100 == 0:
        save_dir = os.path.join('..', 'nn', 'pics')
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, '{}_{}.png'.format(title, y.shape[0])))


# ===== Tee for logging =====
class Tee(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding="utf-8")
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        self.terminal.flush()
        self.log.flush()


# ===== Main Entry =====
import argparse
if __name__ == "__main__":
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sys.stdout = Tee("train_log_{}.txt".format(timestamp))

    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('-m', '--model', help='model path')
    parser.add_argument('-d', '--meta',  help='meta file')
    parser.add_argument('--mode', choices=['original', 'exo'], default='original',
                        help='Training mode: original (default) or exo (exoskeleton)')
    args = parser.parse_args()

    if args.meta is None:
        print('Provide meta file')
        exit()

    print('===== MASS Training [mode={}] ====='.format(args.mode))
    ppo = PPO(args.meta, mode=args.mode)
    nn_dir = '../nn/pics'
    if not os.path.exists(nn_dir):
        os.makedirs(nn_dir)
    if args.model is not None:
        ppo.LoadModel(args.model)
    else:
        ppo.SaveModel()

    print('num states: {}, num actions: {}'.format(ppo.env.GetNumState(), ppo.env.GetNumAction()))
    if args.mode == 'exo':
        print('num exo actions: {}'.format(ppo.num_exo_action))
    for _ in range(ppo.max_iteration - 5):
        ppo.Train()
        rewards = ppo.Evaluate()
        Plot(rewards, 'reward', 0, False)
