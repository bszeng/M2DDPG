import copy
import torch
import torch.nn as nn
import numpy as np
from datetime import datetime

torch.autograd.set_detect_anomaly(True)

class Actor_Feature(nn.Module):
    def __init__(self, state_dim):
        super(Actor_Feature, self).__init__()
        hidden_dim = 1 if state_dim == 0 else 2 ** (state_dim - 1).bit_length()
        self.l1 = nn.Linear(state_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, state):
        x = torch.tanh(self.l1(state.float()))
        x = torch.tanh(self.l2(x))
        return x

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, M, N, K, power_t, device, max_action=1):
        super(Actor, self).__init__()
        hidden_dim = 1 if state_dim == 0 else 2 ** (state_dim - 1).bit_length()
        self.device = device
        self.M = M
        self.N = N
        self.K = K
        self.power_t = power_t
        self.max_action = max_action
        self.hidden_dim = hidden_dim

        self.l3 = nn.Linear(hidden_dim, action_dim)

    def compute_power(self, a):
        # Normalize the power
        G_real = a[:, :self.M ** 2].cpu().data.numpy()
        G_imag = a[:, self.M ** 2:2 * self.M ** 2].cpu().data.numpy()
        G = G_real.reshape(G_real.shape[0], self.M, self.K) + 1j * G_imag.reshape(G_imag.shape[0], self.M, self.K)
        GG_H = np.matmul(G, np.transpose(G.conj(), (0, 2, 1)))
        current_power_t = torch.sqrt(torch.from_numpy(np.real(np.trace(GG_H, axis1=1, axis2=2)))).reshape(-1, 1).to(self.device)
        return current_power_t

    def compute_phase(self, a):
        # Normalize the phase matrix
        Phi_real = a[:, -2 * self.N:-self.N].detach()
        Phi_imag = a[:, -self.N:].detach()
        return torch.sum(torch.abs(Phi_real), dim=1).reshape(-1, 1) * np.sqrt(2), torch.sum(torch.abs(Phi_imag), dim=1).reshape(-1, 1) * np.sqrt(2)

    def forward(self, x):
        a = torch.tanh(self.l3(x).float())
        # Normalize the transmission power and phase matrix
        current_power_t = self.compute_power(a.detach()).expand(-1, 2 * self.M ** 2) / np.sqrt(self.power_t)
        real_normal, imag_normal = self.compute_phase(a.detach())
        real_normal = real_normal.expand(-1, self.N)
        imag_normal = imag_normal.expand(-1, self.N)

        division_term = torch.cat([current_power_t, real_normal, imag_normal], dim=1)
        action = self.max_action * a / division_term

        return action

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        hidden_dim = 1 if (state_dim + action_dim) == 0 else 2 ** ((state_dim + action_dim) - 1).bit_length()
        self.l1 = nn.Linear(state_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim + action_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        q = torch.tanh(self.l1(state.float()))
        q = torch.tanh(self.l2(torch.cat([q, action], 1)))
        q = self.l3(q)
        return q

class DDPG(object):
    def __init__(self, state_dim, action_dim, M, N, K, power_t, max_action, actor_lr, critic_lr, actor_decay, critic_decay, device, discount=0.99, tau=0.001):
        self.device = device
        self.lr = critic_lr
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.actor_decay = actor_decay
        self.critic_decay = critic_decay
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.M = M
        self.N = N
        self.K = K
        self.tau = tau

        powert_t_W = 10 ** (power_t / 10)

        self.meta_lr= 0.0001
        self.inner_lr= 0.0001

        # Initialize actor networks and optimizer
        self.actor_feature = Actor_Feature(state_dim).to(self.device)
        self.actor = Actor(state_dim, action_dim, M, N, K, powert_t_W, max_action=max_action, device=device).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr, weight_decay=self.actor_decay)

        self.actor_feature_target = copy.deepcopy(self.actor_feature)
        self.actor_feature_optimizer = torch.optim.Adam(
            self.actor_feature.parameters(), 
            lr=self.actor_lr * 1.0 , 
            weight_decay=self.actor_decay)

        # Initialize critic networks and optimizer
        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr, weight_decay=self.critic_decay)
        
        self.meta_actor_optimizer = torch.optim.Adam(
            list(self.actor_feature.parameters()) + list(self.actor.parameters()), lr=self.meta_lr
        )
        self.meta_critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.meta_lr)

        self.inner_actor_optimizer = torch.optim.Adam(
            list(self.actor_feature.parameters()) + list(self.actor.parameters()), lr=self.inner_lr
        )
        self.inner_critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.inner_lr)

        self.critic_loss_fn = nn.MSELoss()

        # Initialize the discount and target update rated
        self.discount = discount
        self.tau = tau

    def select_action(self, state):
        self.actor_feature.eval()
        self.actor.eval()
        with torch.no_grad():
            state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
            state_feature = self.actor_feature(state)
            action = self.actor(state_feature).cpu().data.numpy().flatten().reshape(1, -1)
        return action

    def save(self, file_name):
        torch.save(self.critic.state_dict(), file_name + "_critic")
        torch.save(self.critic_target.state_dict(), file_name + "_critic_target")        
        torch.save(self.actor_feature.state_dict(), file_name + "_actor_feature")
        torch.save(self.actor_feature_target.state_dict(), file_name + "_actor_feature_target")        
        torch.save(self.actor.state_dict(), file_name + "_actor")
        torch.save(self.actor_target.state_dict(), file_name + "_actor_target")

        torch.save(self.actor_optimizer.state_dict(), file_name + "_actor_optimizer")
        torch.save(self.critic_optimizer.state_dict(), file_name + "_critic_optimizer")        
        torch.save(self.meta_actor_optimizer.state_dict(), file_name + "_meta_actor_optimizer")
        torch.save(self.meta_critic_optimizer.state_dict(), file_name + "_meta_critic_optimizer")        
        torch.save(self.inner_actor_optimizer.state_dict(), file_name + "_inner_actor_optimizer")
        torch.save(self.inner_critic_optimizer.state_dict(), file_name + "_inner_critic_optimizer")

    def load(self, file_name):

        map_location = self.device

        self.critic.load_state_dict(torch.load(file_name + "_critic", map_location=map_location))
        self.critic_target.load_state_dict(torch.load(file_name + "_critic_target", map_location=map_location))        
        self.actor_feature.load_state_dict(torch.load(file_name + "_actor_feature", map_location=map_location))
        self.actor_feature_target.load_state_dict(torch.load(file_name + "_actor_feature_target", map_location=map_location))        
        self.actor.load_state_dict(torch.load(file_name + "_actor", map_location=map_location))
        self.actor_target.load_state_dict(torch.load(file_name + "_actor_target", map_location=map_location))

        self.actor_optimizer.load_state_dict(torch.load(file_name + "_actor_optimizer", map_location=map_location))
        self.critic_optimizer.load_state_dict(torch.load(file_name + "_critic_optimizer", map_location=map_location))        
        self.meta_actor_optimizer.load_state_dict(torch.load(file_name + "_meta_actor_optimizer", map_location=map_location))
        self.meta_critic_optimizer.load_state_dict(torch.load(file_name + "_meta_critic_optimizer", map_location=map_location))        
        self.inner_actor_optimizer.load_state_dict(torch.load(file_name + "_inner_actor_optimizer", map_location=map_location))
        self.inner_critic_optimizer.load_state_dict(torch.load(file_name + "_inner_critic_optimizer", map_location=map_location))

    def inner_update(self, restored_buffer, task_idx, inner_steps, batch_size=64):

        initial_feat_params = copy.deepcopy([p.data for p in self.actor_feature.parameters()])
        initial_actor_params = copy.deepcopy([p.data for p in self.actor.parameters()])
        initial_critic_params = copy.deepcopy([p.data for p in self.critic.parameters()])
        
        self.actor_feature.train()
        self.actor.train()
        self.critic.train()

        actor_losses = []
        critic_losses = []
        
        for step in range(inner_steps):

            state, action, next_state, reward, not_done = restored_buffer.sample(task_idx, batch_size)
            
            with torch.no_grad():
                next_feat = self.actor_feature_target(next_state)
                next_action = self.actor_target(next_feat)
                target_q = self.critic_target(next_state, next_action)
                target_q = reward + not_done * self.discount * target_q

            current_q = self.critic(state, action)
            critic_loss = self.critic_loss_fn(current_q, target_q)            
            self.inner_critic_optimizer.zero_grad()
            critic_loss.backward()
            self.inner_critic_optimizer.step()
            critic_losses.append(critic_loss.item())
            
            feat = self.actor_feature(state)
            actor_action = self.actor(feat)
            actor_loss = -self.critic(state, actor_action).mean()            
            self.inner_actor_optimizer.zero_grad()
            actor_loss.backward()
            self.inner_actor_optimizer.step()
            actor_losses.append(actor_loss.item())

        if step % 100 == 0:
            print(f"[Task {task_idx}] Inner Step {step} | Actor Loss: {actor_loss.item():.4f} | Critic Loss: {critic_loss.item():.4f}")
        
        return initial_feat_params, initial_actor_params, initial_critic_params, actor_losses, critic_losses
    
    def meta_train(self, restored_buffer, meta_steps, inner_steps, task_batch_size, num_tasks):
        meta_actor_losses = []
        meta_critic_losses = []

        for meta_step in range(meta_steps):

            feat_param_updates = []
            actor_param_updates = []
            critic_param_updates = []
            task_actor_losses = []
            task_critic_losses = []

            for _ in range(task_batch_size):
                task_idx = np.random.randint(0, num_tasks)
                
                init_feat_params, init_actor_params, init_critic_params, actor_loss_list, critic_loss_list = self.inner_update(
                    restored_buffer, task_idx, inner_steps
                )

                feat_delta = [p.data - init_p for p, init_p in zip(self.actor_feature.parameters(), init_feat_params)]
                actor_delta = [p.data - init_p for p, init_p in zip(self.actor.parameters(), init_actor_params)]
                critic_delta = [p.data - init_p for p, init_p in zip(self.critic.parameters(), init_critic_params)]
                
                feat_param_updates.append(feat_delta)
                actor_param_updates.append(actor_delta)
                critic_param_updates.append(critic_delta)
                task_actor_losses.append(np.mean(actor_loss_list))
                task_critic_losses.append(np.mean(critic_loss_list))

                for p, init_p in zip(self.actor_feature.parameters(), init_feat_params):
                    p.data.copy_(init_p)
                for p, init_p in zip(self.actor.parameters(), init_actor_params):
                    p.data.copy_(init_p)
                for p, init_p in zip(self.critic.parameters(), init_critic_params):
                    p.data.copy_(init_p)

            if feat_param_updates and actor_param_updates and critic_param_updates:
                avg_feat_delta = [torch.mean(torch.stack([delta[i] for delta in feat_param_updates]), dim=0) 
                                for i in range(len(feat_param_updates[0]))]
                avg_actor_delta = [torch.mean(torch.stack([delta[i] for delta in actor_param_updates]), dim=0) 
                                for i in range(len(actor_param_updates[0]))]
                avg_critic_delta = [torch.mean(torch.stack([delta[i] for delta in critic_param_updates]), dim=0) 
                                    for i in range(len(critic_param_updates[0]))]

                self.meta_actor_optimizer.zero_grad()
                for p, delta in zip(self.actor_feature.parameters(), avg_feat_delta):
                    p.grad = -delta / self.meta_lr
                for p, delta in zip(self.actor.parameters(), avg_actor_delta):
                    p.grad = -delta / self.meta_lr
                self.meta_actor_optimizer.step()

                self.meta_critic_optimizer.zero_grad()
                for p, delta in zip(self.critic.parameters(), avg_critic_delta):
                    p.grad = -delta / self.meta_lr
                self.meta_critic_optimizer.step()

                if meta_step % 10 == 0:
                    avg_actor_loss = np.mean(task_actor_losses) if task_actor_losses else 0.0
                    avg_critic_loss = np.mean(task_critic_losses) if task_critic_losses else 0.0
                    
                    meta_actor_losses.append(avg_actor_loss)
                    meta_critic_losses.append(avg_critic_loss)
                    
                    print(f"[{datetime.now()}] Outter:{meta_steps}-Inner:{inner_steps}-Tbs:{task_batch_size} | "
                        f"Outter Step {meta_step} | "
                        f"Avg Actor Loss: {avg_actor_loss:.6f} | "
                        f"Avg Critic Loss: {avg_critic_loss:.6f} ")

            if meta_step % 5 == 0:
                self.soft_update(self.critic_target, self.critic)
                self.soft_update(self.actor_feature_target, self.actor_feature)
                self.soft_update(self.actor_target, self.actor)

        return meta_actor_losses, meta_critic_losses

    def soft_update(self, target, source):
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                self.tau * source_param.data + (1 - self.tau) * target_param.data
            )