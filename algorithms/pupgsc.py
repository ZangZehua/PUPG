import os.path
import time
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import geomloss

from .network import NatureCNN, NonLinearQNet
from .utils import ReplayBuffer, LinearSchedule, prepare_obs, RandomShiftsAug, off_diagonal


class Network(nn.Module):
    def __init__(self, args, in_depth, action_shape):
        super().__init__()
        self.backbone = NatureCNN(in_depth)
        self.q_network = NonLinearQNet(self.backbone.out_features, args.q_hidden_features, action_shape,
                                       args.v_min, args.v_max, args.n_atoms, args.noisy_std, args.log_softmax)

    def forward(self, x, action=None):
        features = self.backbone(x)
        actions, logit = self.q_network(features, action)
        return actions, logit

    def reset_noise(self):
        self.q_network.reset_noise()

    def get_q_dist(self, x):
        features = self.backbone(x)
        q_dist = self.q_network.get_q_dist(features)
        return q_dist

    def get_feature(self, x):
        features = self.backbone(x)
        return features


class Policy:
    def __init__(self, args, in_depth, action_space, device):
        self.args = args
        self.action_shape = action_space.n
        self.device = device
        self.q_policy = Network(args, in_depth, self.action_shape).to(self.device)
        self.q_target = Network(args, in_depth, self.action_shape).to(self.device)

        self.optimizer = optim.Adam(self.q_policy.parameters(), lr=self.args.learning_rate,
                                    eps=self.args.eps, betas=(0.9, 0.999))
        self.update_target()
        self.waug = RandomShiftsAug(self.args.padding)
        self.loss_fn = geomloss.SamplesLoss(
            loss='sinkhorn', p=args.p,
            # 对于p=1或p=2的情形
            cost=geomloss.utils.distances if args.p == 1 else geomloss.utils.squared_distances,
            blur=.1 ** (1 / args.p), backend='tensorized')

    def select_action(self, obs):
        actions, _ = self.q_policy(prepare_obs(obs).to(self.device))
        actions = actions.cpu().numpy()
        return actions

    def learn(self, data):
        loss_rainbow, q_values, before_mean_loss = self.learn_rainbow(data)
        loss_cl, weight = self.learn_cl(data)
        loss = loss_rainbow + self.args.cl_coef * loss_cl
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        logs = {
            "losses/loss": loss.item(),
            "losses/loss_rainbow": loss_rainbow.item(),
            "losses/q_values": q_values,
            "losses/loss_cl": loss_cl.item(),
            "losses/weight": weight,
        }
        return logs, before_mean_loss

    def learn_rainbow(self, data):
        obs, next_obs, actions, rewards, dones = (
            data['obs'], data['next_obs'], data['action'], data['reward'], data['done'])
        with torch.no_grad():
            next_actions, _ = self.q_policy(next_obs)
            _, next_pmfs = self.q_target(next_obs, next_actions)
            # _, next_pmfs = self.q_target(next_obs)
            next_atoms = rewards + pow(self.args.gamma, self.args.n_steps) * self.q_target.q_network.atoms * (~dones)
            # projection
            delta_z = self.q_target.q_network.atoms[1] - self.q_target.q_network.atoms[0]
            tz = next_atoms.clamp(self.args.v_min, self.args.v_max)

            b = (tz - self.args.v_min) / delta_z
            l = b.floor().clamp(0, self.args.n_atoms - 1)
            u = b.ceil().clamp(0, self.args.n_atoms - 1)
            # (l == u).float() handles the case where bj is exactly an integer
            # example bj = 1, then the upper ceiling should be uj= 2, and lj= 1
            d_m_l = (u + (l == u).float() - b) * next_pmfs
            d_m_u = (b - l) * next_pmfs
            target_pmfs = torch.zeros_like(next_pmfs)
            for i in range(target_pmfs.size(0)):
                target_pmfs[i].index_add_(0, l[i].long(), d_m_l[i])
                target_pmfs[i].index_add_(0, u[i].long(), d_m_u[i])
        _, old_pmfs = self.q_policy(obs, actions.squeeze(-1))
        before_mean_loss = (-(target_pmfs * old_pmfs.clamp(min=1e-5, max=10 - 1e-5).log()).sum(-1))
        loss = before_mean_loss.mean()
        old_val = (old_pmfs * self.q_policy.q_network.atoms).sum(1)
        return loss, old_val.mean().item(), before_mean_loss.detach().cpu().numpy()

    def learn_cl(self, data):
        obs = data['obs']
        obs_a = obs
        obs_b = self.waug(obs.float())

        with torch.no_grad():
            z1 = self.q_policy.get_feature(obs_a)
            z2 = self.q_policy.get_feature(obs_b)
            c = z1.T @ z2
            on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
            off_diag = off_diagonal(c).pow_(2).sum()
            weight = on_diag + self.args.lambd * off_diag
            weight = weight.mean().item()

        q_a = self.q_policy.get_q_dist(obs_a)
        q_b = self.q_policy.get_q_dist(obs_b)
        pw = self.loss_fn(q_a, q_b)
        # TODO: coef add linear scheduling
        pw = self.args.bt_coef * weight * pw
        return pw, weight

    def update_target(self):
        self.q_target.load_state_dict(self.q_policy.state_dict())

    def reset_noise(self):
        self.q_policy.reset_noise()

    def save_model(self, save_path):
        torch.save(self.q_policy.state_dict(), save_path)

    def load_model(self, model_path):
        self.q_policy.load_state_dict(torch.load(model_path))


class Rainbow:
    def __init__(self, args, envs, eval_env, device):
        self.paths = args
        self.model_list = self.paths.model_list
        self.args = args.algo
        self.envs = envs
        self.eval_env = eval_env
        self.device = device
        print(self.envs.single_observation_space)
        print(self.envs.single_action_space)
        print(self.device)
        print("============================================================")

        self.writer = SummaryWriter(self.paths.save_path, flush_secs=2)
        self.policy = Policy(self.args, self.envs.single_observation_space.shape[0], self.envs.single_action_space,
                             self.device)
        self.buffer = ReplayBuffer(self.args.buffer_size,
                                   self.envs.single_observation_space.shape,
                                   self.args.gamma,
                                   self.args.alpha,
                                   self.args.n_steps,
                                   self.device)

    @torch.no_grad()
    def eval(self):
        obs, _ = self.eval_env.reset()
        while True:
            action = self.policy.select_action(obs)
            obs, _, _, _, infos = self.eval_env.step(action)
            if "final_info" in infos:
                for info in infos["final_info"]:
                    # Skip the envs that are not done
                    if info is None or "episode" not in info:
                        continue
                    return info['episode']['r']

    def run(self):
        per_beta_schedule = LinearSchedule(0, initial_value=self.args.beta_s, final_value=1.0,
                                           decay_time=self.args.max_steps)
        epsilon = LinearSchedule(0, initial_value=self.args.epsilon, final_value=0.001,
                                 decay_time=self.args.max_steps)
        global_step = 0
        eval_times = 0
        training = True
        start_time = time.time()
        obs, _ = self.envs.reset()
        while training:
            if random.random() < epsilon(global_step):
                actions = self.envs.action_space.sample()
            else:
                actions = self.policy.select_action(obs)
            next_obs, rewards, terminated, truncated, infos = self.envs.step(actions)
            if "final_info" in infos:
                for info in infos["final_info"]:
                    # Skip the envs that are not done
                    if info is None or "episode" not in info:
                        continue
                    print(f"global_step={global_step}, "
                          f"episodic_reward={info['episode']['r']}, "
                          f"episodic_length={info['episode']['l']}, "
                          f"time_used={np.round((time.time() - start_time), 2)}")
                    self.writer.add_scalar("charts/episodic_reward", info["episode"]["r"], global_step)
                    self.writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                    if global_step >= self.args.max_steps:
                        training = False
            real_next_obs = next_obs.copy()
            for idx, d in enumerate(truncated):
                if d:
                    real_next_obs[idx] = infos["final_observation"][idx]
            self.buffer.add(obs, real_next_obs, actions, rewards, terminated)
            obs = next_obs
            if global_step % self.args.train_freq == 0:
                self.policy.reset_noise()
            global_step += 1

            if global_step >= self.args.learning_starts:
                if global_step % self.args.train_freq == 0:
                    beta = per_beta_schedule(global_step)
                    self.writer.add_scalar("charts/beta", beta, global_step)
                    data = self.buffer.sample(self.args.batch_size, beta)
                    logs, um_loss = self.policy.learn(data)
                    new_priorities = np.abs(um_loss) + 1e-6
                    # Update replay buffer priorities
                    self.buffer.update_priorities(data['indexes'], new_priorities)
                    if global_step % 100 == 0:
                        for k, v in logs.items():
                            self.writer.add_scalar(k, v, global_step)
                        self.writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
                if global_step % self.args.target_network_freq == 0:
                    self.policy.update_target()
                if (len(self.model_list) > 0 and eval_times < len(self.model_list)
                        and global_step >= self.model_list[eval_times]):
                    eval_times += 1
                    print("============================================================")
                    print("eval")
                    eval_rewards = []
                    eval_path = open(os.path.join(self.paths.save_path, "eval" + str(global_step) + ".csv"), "a+")
                    for _ in range(self.args.eval_times):
                        er = self.eval()
                        print(er)
                        eval_rewards.append(er[0])
                        eval_path.write(str(er[0]) + "\n")
                        eval_path.flush()
                    print("avg reward:", np.mean(eval_rewards))
                    eval_path.close()
