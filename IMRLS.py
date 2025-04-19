# -*- coding: utf-8 -*-
# الكود الكامل للنظام المتكامل (IMRLS - Integrated Mathematical RL System)

"""
IMRLS: نظام تعلم معزز مبتكر يعتمد على معادلات رياضية تتطوّر مع التدريب
تم تطويره بواسطة: [باسل يحيى عبدالله/ العراق/ الموصل ]
تاريخ الإصدار: [20/4/2025]
"""

import torch
import torch.nn as nn
import torch.optim as optim  # <--- استيراد optim
import numpy as np

# --- تغيير الواجهة الخلفية لـ Matplotlib ---
import matplotlib
try:
    # محاولة استخدام Agg (للحفظ فقط)
    matplotlib.use('Agg')
    print("Matplotlib backend set to Agg.")
except Exception as e: # استخدام Exception أعم للقبض على أي خطأ محتمل
    print(f"Warning: Failed to set Matplotlib backend to Agg. Error: {e}")
# --- نهاية تغيير الواجهة الخلفية ---

import matplotlib.pyplot as plt # الآن استيراد pyplot
from collections import deque
from copy import deepcopy
import random                 # <--- استيراد random
import os
import json

# --- المكونات الأساسية للنظام ---

class DynamicMathUnit(nn.Module):
    """
    وحدة رياضية تمثل معادلة ديناميكية قابلة للتعلم.
    (نفس الكود المستخدم في النسخة النهائية المستقرة لـ EvoEqNet)
    """
    def __init__(self, input_dim, output_dim, complexity=5):
        super().__init__()
        if not (isinstance(input_dim, int) and input_dim > 0 and
                isinstance(output_dim, int) and output_dim > 0 and
                isinstance(complexity, int) and complexity > 0):
            raise ValueError("input_dim, output_dim, and complexity must be positive integers.")
        self.input_dim = input_dim; self.output_dim = output_dim; self.complexity = complexity
        self.internal_dim = max(output_dim, complexity, input_dim // 2 + 1)
        self.input_layer = nn.Linear(input_dim, self.internal_dim)
        self.output_layer = nn.Linear(self.internal_dim, output_dim)
        self.layer_norm = nn.LayerNorm(self.internal_dim)
        self.coeffs = nn.Parameter(torch.randn(self.internal_dim, self.complexity) * 0.05)
        self.exponents = nn.Parameter(torch.rand(self.internal_dim, self.complexity) * 1.5 + 0.25)
        self.base_funcs = [
            torch.sin, torch.cos, torch.tanh, nn.SiLU(), nn.ReLU6(),
            lambda x: torch.sigmoid(x) * x,
            lambda x: 0.5 * x * (1 + torch.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * torch.pow(x, 3))))
        ]
        self.num_base_funcs = len(self.base_funcs)
        nn.init.xavier_uniform_(self.input_layer.weight); nn.init.zeros_(self.input_layer.bias)
        nn.init.xavier_uniform_(self.output_layer.weight); nn.init.zeros_(self.output_layer.bias)

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
             try: x = torch.tensor(x, dtype=torch.float32, device=self.input_layer.weight.device)
             except Exception as e: raise TypeError(f"Input conversion failed: {e}")
        if x.dim() == 1: x = x.unsqueeze(0)
        if x.dim() == 0: x = x.unsqueeze(0).unsqueeze(0).expand(1, self.input_dim)
        elif x.shape[1] != self.input_dim:
            if x.numel() == x.shape[0] * self.input_dim: x = x.view(x.shape[0], self.input_dim)
            else: raise ValueError(f"{self.__class__.__name__} expects input_dim={self.input_dim}, got shape {x.shape}")
        internal_x = self.input_layer(x); internal_x = self.layer_norm(internal_x); internal_x = torch.relu(internal_x)
        dynamic_sum = torch.zeros_like(internal_x)
        for i in range(self.complexity):
            func = self.base_funcs[i % self.num_base_funcs]
            coeff_i = self.coeffs[:, i].unsqueeze(0); exp_i = self.exponents[:, i].unsqueeze(0)
            term_input = internal_x * exp_i; term_input_clamped = torch.clamp(term_input, -10.0, 10.0)
            try:
                term = coeff_i * func(term_input_clamped)
                term = torch.nan_to_num(term, nan=0.0, posinf=1e4, neginf=-1e4)
                dynamic_sum = dynamic_sum + term
            except RuntimeError as e: continue
        output = self.output_layer(dynamic_sum); output = torch.clamp(output, -100.0, 100.0)
        return output

class TauRLayer(nn.Module):
    """
    طبقة تحسب قيمة Tau التي توازن بين التقدم والمخاطر.
    (نفس الكود المستخدم في النسخة النهائية المستقرة لـ MathRLSystem)
    """
    def __init__(self, input_dim, output_dim, epsilon=1e-6, alpha=0.1, beta=0.1):
        super().__init__()
        if not (isinstance(input_dim, int) and input_dim > 0 and isinstance(output_dim, int) and output_dim > 0):
             raise ValueError("input_dim and output_dim must be positive integers.")
        self.progress_transform = nn.Linear(input_dim, output_dim)
        self.risk_transform = nn.Linear(input_dim, output_dim)
        nn.init.xavier_uniform_(self.progress_transform.weight, gain=0.1); nn.init.zeros_(self.progress_transform.bias)
        nn.init.xavier_uniform_(self.risk_transform.weight, gain=0.1); nn.init.zeros_(self.risk_transform.bias)
        self.epsilon = epsilon; self.alpha = alpha; self.beta = beta; self.min_denominator = 1e-5

    def forward(self, x):
        if not isinstance(x, torch.Tensor): x = torch.tensor(x, dtype=torch.float32, device=self.progress_transform.weight.device)
        if x.dim() == 1: x = x.unsqueeze(0)
        if x.dim() == 0: x = x.unsqueeze(0).unsqueeze(0).expand(1, self.progress_transform.in_features)
        if x.shape[1] != self.progress_transform.in_features: raise ValueError(f"{self.__class__.__name__} expects input_dim={self.progress_transform.in_features}, got shape {x.shape}")
        progress = torch.tanh(self.progress_transform(x)); risk = torch.relu(self.risk_transform(x))
        numerator = progress + self.alpha; denominator = risk + self.beta + self.epsilon
        denominator = torch.clamp(denominator, min=self.min_denominator); tau_output = numerator / denominator
        tau_output = torch.tanh(tau_output); tau_output = torch.clamp(tau_output, min=-10.0, max=10.0)
        return tau_output

class ChaosOptimizer(optim.Optimizer):
    # --- (الكود كما هو في الرد السابق) ---
    def __init__(self, params, lr=0.001, sigma=10.0, rho=28.0, beta=8/3):
        if lr < 0.0: raise ValueError(f"Invalid learning rate: {lr}")
        defaults = dict(lr=lr, sigma=sigma, rho=rho, beta=beta); super().__init__(params, defaults)
    @torch.no_grad()
    def step(self, closure=None):
        loss = None;
        if closure is not None:
            with torch.enable_grad(): loss = closure()
        for group in self.param_groups:
            lr, sigma, rho, beta_chaos = group['lr'], group['sigma'], group['rho'], group['beta']
            if not group['params']: continue
            for p in group['params']:
                if p.grad is None: continue
                grad = p.grad; param_state = p.data
                try:
                    dx = sigma * (grad - param_state); dy = param_state * (rho - grad) - param_state
                    dz = param_state * grad - beta_chaos * param_state; chaotic_update = dx + dy + dz
                    if not torch.isfinite(chaotic_update).all(): continue
                    p.data.add_(chaotic_update, alpha=lr)
                except RuntimeError as e: continue
        return loss

class IntegratedEvolvingNetwork(nn.Module):
    # --- (الكود كما هو في الرد السابق، مع آلية add_layer المصححة) ---
    def __init__(self, input_dim, hidden_dims, output_dim, use_dynamic_units=False, max_layers=8):
        super().__init__(); self.input_dim = input_dim; self.hidden_dims = list(hidden_dims); self.output_dim = output_dim
        self.use_dynamic_units = use_dynamic_units; self.max_layers = max_layers; self.layers = nn.ModuleList()
        current_dim = input_dim
        for i, hidden_dim in enumerate(self.hidden_dims):
            if not (isinstance(current_dim, int) and current_dim > 0 and isinstance(hidden_dim, int) and hidden_dim > 0): raise ValueError(f"Invalid dims layer {i}")
            layer_modules = []; block_input_dim = current_dim
            if use_dynamic_units: layer_modules.append(DynamicMathUnit(current_dim, current_dim)); block_input_dim = current_dim
            layer_modules.extend([nn.Linear(block_input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(), TauRLayer(hidden_dim, hidden_dim)])
            self.layers.append(nn.Sequential(*layer_modules)); current_dim = hidden_dim
        self.output_layer = nn.Linear(current_dim, output_dim)
        self.performance_history = deque(maxlen=50); self.layer_evolution_threshold = 0.002
    def get_architecture_info(self):
        current_hidden_dims = [layer[-1].progress_transform.in_features for layer in self.layers] if self.layers else []
        info = {'input_dim': self.input_dim, 'hidden_dims': current_hidden_dims, 'output_dim': self.output_dim,
                'use_dynamic_units': self.use_dynamic_units, 'max_layers': self.max_layers,}
        return info
    @classmethod
    def from_architecture_info(cls, info):
        net = cls(info['input_dim'], info['hidden_dims'], info['output_dim'], info.get('use_dynamic_units', False), info.get('max_layers', 8))
        return net
    def add_layer(self):
        if len(self.layers) >= self.max_layers: return False
        print(f"*** Evolving IMRLS Network: Adding hidden layer {len(self.layers) + 1}/{self.max_layers} ***")
        try: device = next(self.parameters()).device
        except StopIteration: device = torch.device("cpu")
        if self.layers: last_layer_output_dim = self.layers[-1][-1].progress_transform.in_features
        else: last_layer_output_dim = self.input_dim
        new_layer_input_dim = last_layer_output_dim
        # استخدام بعد ثابت للطبقات الجديدة أو آخر بعد مخفي
        new_layer_hidden_dim = self.hidden_dims[-1] if self.hidden_dims else max(32, self.output_dim) # fallback dimension

        if not isinstance(new_layer_input_dim, int) or new_layer_input_dim <= 0: print(f"Error: Invalid input dim ({new_layer_input_dim})"); return False
        if not isinstance(new_layer_hidden_dim, int) or new_layer_hidden_dim <= 0: print(f"Error: Invalid hidden dim ({new_layer_hidden_dim})"); return False
        new_layer_modules = []; current_dim_new = new_layer_input_dim
        if self.use_dynamic_units: new_layer_modules.append(DynamicMathUnit(current_dim_new, current_dim_new))
        new_layer_modules.extend([nn.Linear(current_dim_new, new_layer_hidden_dim), nn.LayerNorm(new_layer_hidden_dim), nn.ReLU(), TauRLayer(new_layer_hidden_dim, new_layer_hidden_dim)])
        new_sequential_layer = nn.Sequential(*new_layer_modules).to(device)
        self.layers.append(new_sequential_layer)
        # تحديث hidden_dims للتتبع فقط
        self.hidden_dims.append(new_layer_hidden_dim); print(f"Current hidden dimensions trace: {self.hidden_dims}")
        print(f"Rebuilding output layer to accept input dim: {new_layer_hidden_dim}")
        self.output_layer = nn.Linear(new_layer_hidden_dim, self.output_dim).to(device)
        nn.init.xavier_uniform_(self.output_layer.weight); nn.init.zeros_(self.output_layer.bias)
        return True
    def evolve_structure(self, validation_metric):
        evolved = False;
        if not np.isfinite(validation_metric): return evolved
        self.performance_history.append(validation_metric)
        if len(self.performance_history) < self.performance_history.maxlen: return evolved
        recent_metrics = list(self.performance_history)
        if len(recent_metrics) > 20:
            improvement = np.mean(recent_metrics[-10:]) - np.mean(recent_metrics[-20:-10])
            if improvement < self.layer_evolution_threshold:
                if self.add_layer(): evolved = True; self.performance_history.clear()
        return evolved
    def forward(self, x):
        if not isinstance(x, torch.Tensor): x = torch.tensor(x, dtype=torch.float32, device=next(self.parameters()).device)
        if x.dim() == 1: x = x.unsqueeze(0)
        if x.shape[1] != self.input_dim:
             if x.numel() == x.shape[0] * self.input_dim: x = x.view(x.shape[0], self.input_dim)
             else: raise ValueError(f"{self.__class__.__name__} expects input_dim={self.input_dim}, got {x.shape}")
        current_x = x
        if not self.layers: return self.output_layer(current_x)
        for i, layer_block in enumerate(self.layers): current_x = layer_block(current_x)
        output = self.output_layer(current_x)
        return output

# --- نظام التدريب المتكامل ---
class IMRLS_Trainer:
    """
    نظام تدريب متكامل للشبكة التطورية الرياضية في بيئة تعلم معزز (IMRLS).
    """
    def __init__(self, input_dim, action_dim, hidden_dims,
                 use_dynamic_units=False, use_chaos_optimizer=False,
                 learning_rate=0.0005, gamma=0.99, buffer_size=100000,
                 batch_size=64, update_target_every=15, tau_update=True,
                 epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.997):

        self.state_dim = input_dim; self.action_dim = action_dim; self.gamma = gamma
        self.batch_size = batch_size; self.update_target_every = update_target_every
        self.tau_update = tau_update; self.memory = deque(maxlen=buffer_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); print(f"Initializing IMRLS Trainer on device: {self.device}")
        self.initial_arch_info = {'input_dim': input_dim, 'hidden_dims': list(hidden_dims), 'output_dim': action_dim, 'use_dynamic_units': use_dynamic_units}
        self.use_chaos_optimizer = use_chaos_optimizer
        self.net = IntegratedEvolvingNetwork(input_dim, list(hidden_dims), action_dim, use_dynamic_units).to(self.device)
        self.target_net = deepcopy(self.net).to(self.device); self.target_net.eval()
        self.optimizer = None; self._rebuild_optimizer(learning_rate)
        self.loss_fn = nn.MSELoss(); self.update_target_counter = 0
        self.epsilon = epsilon_start; self.epsilon_end = epsilon_end; self.epsilon_decay = epsilon_decay
        self.model_save_path = 'best_imrls_model.pth'; self.arch_save_path = 'best_imrls_arch.json'
        self.best_avg_reward = -float('inf')

    def _rebuild_optimizer(self, current_lr):
        print(f"Rebuilding optimizer with LR: {current_lr}...")
        try:
            current_params = list(self.net.parameters())
            if not any(p.requires_grad for p in current_params): print("Warning: No trainable parameters."); return False
            if self.use_chaos_optimizer: self.optimizer = ChaosOptimizer(current_params, lr=current_lr)
            else: self.optimizer = optim.AdamW(current_params, lr=current_lr, weight_decay=1e-4)
            print(f"Optimizer rebuilt ({'Chaos' if self.use_chaos_optimizer else 'AdamW'}).")
            return True
        except Exception as e: print(f"Unexpected error rebuilding optimizer: {e}"); return False

    def calculate_tau(self, reward):
        progress = max(reward, 0); risk = abs(min(reward, 0))
        tau = (progress + 0.1) / (risk + 0.1 + 1e-8); return np.clip(tau, 0, 100)

    def adaptive_exploration(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def select_action(self, state):
        if random.random() < self.epsilon: return random.randrange(self.action_dim)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device); self.net.eval()
            with torch.no_grad(): q_values = self.net(state_tensor)
            self.net.train(); return torch.argmax(q_values).item()

    def store_transition(self, state, action, reward, next_state, done):
        state = np.array(state, dtype=np.float32); next_state = np.array(next_state, dtype=np.float32)
        self.memory.append((state, action, float(reward), next_state, bool(done)))

    def _update_target_network_weights(self):
        try: self.target_net.load_state_dict(self.net.state_dict())
        except RuntimeError as e:
            print(f"Error updating target net weights: {e}. Recreating target net.")
            try: self.target_net = deepcopy(self.net); self.target_net.eval(); print("Target network rebuilt.")
            except Exception as deepcopy_e: print(f"FATAL: Failed to rebuild target net: {deepcopy_e}")

    def update_network(self):
        if len(self.memory) < self.batch_size: return False
        if self.optimizer is None: print("Error: Optimizer not initialized."); return False
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones_bool = zip(*batch)
        states = torch.FloatTensor(np.array(states)).to(self.device); actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards_orig = torch.FloatTensor(rewards).unsqueeze(1).to(self.device); next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(np.array(dones_bool).astype(np.float32)).unsqueeze(1).to(self.device)
        if self.tau_update: target_rewards = torch.FloatTensor([self.calculate_tau(r) for r in rewards]).unsqueeze(1).to(self.device)
        else: target_rewards = rewards_orig
        self.net.train(); current_q_values = self.net(states); current_q = torch.gather(current_q_values, 1, actions)
        self.target_net.eval()
        with torch.no_grad(): next_q_values_target = self.target_net(next_states); next_q_target = next_q_values_target.max(1)[0].unsqueeze(1)
        expected_q = target_rewards + (1.0 - dones) * self.gamma * next_q_target
        loss = self.loss_fn(current_q, expected_q)
        if not torch.isfinite(loss): print(f"Warning: Non-finite loss ({loss.item()}). Skipping."); return False
        self.optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0); self.optimizer.step()
        structure_changed = self.net.evolve_structure(loss.item())
        if structure_changed:
             current_lr = self.optimizer.param_groups[0]['lr']
             if self._rebuild_optimizer(current_lr):
                  print("Recreating target network after structure evolution.")
                  try: self.target_net = deepcopy(self.net); self.target_net.eval(); self.update_target_counter = 0
                  except Exception as e: print(f"Error deepcopying evolved network: {e}")
             else: print("ERROR: Failed rebuild optimizer after evolution.")
        else:
            self.update_target_counter += 1
            if self.update_target_counter % self.update_target_every == 0: self._update_target_network_weights()
        return True

    def train(self, env, episodes=1000, max_steps_per_episode=500, save_best_model=True, action_wrapper=None):
        rewards_history = []; total_steps = 0
        print(f"Starting IMRLS training for {episodes} episodes...")
        for ep in range(episodes):
            state = env.reset();
            if not isinstance(state, np.ndarray): state = np.array(state, dtype=np.float32)
            total_reward = 0; episode_steps = 0; done = False; info = {}
            while not done and episode_steps < max_steps_per_episode:
                action = self.select_action(state)
                if action_wrapper: next_state, reward, done, info = action_wrapper(env, action)
                else: next_state, reward, done, info = env.step(action)
                if not isinstance(next_state, np.ndarray): next_state = np.array(next_state, dtype=np.float32)
                self.store_transition(state, action, reward, next_state, done)
                updated = False
                if len(self.memory) >= self.batch_size and total_steps % 4 == 0: updated = self.update_network()
                state = next_state; total_reward += reward; episode_steps += 1; total_steps += 1
            self.adaptive_exploration()
            rewards_history.append(total_reward)
            current_avg_reward = np.mean(rewards_history[-50:]) if len(rewards_history) >= 50 else np.mean(rewards_history) if rewards_history else 0
            if save_best_model and len(rewards_history) > 50 and current_avg_reward > self.best_avg_reward:
                print(f"*** New best avg reward: {current_avg_reward:.2f} (prev {self.best_avg_reward:.2f}). Saving... ***")
                self.best_avg_reward = current_avg_reward
                try:
                     arch_info = self.net.get_architecture_info();
                     with open(self.arch_save_path, 'w') as f: json.dump(arch_info, f)
                     torch.save(self.net.state_dict(), self.model_save_path)
                except Exception as save_e: print(f"Error saving best model: {save_e}")
            if (ep + 1) % 20 == 0:
                 reason = info.get('reason', 'max_steps') if done else 'in_prog'
                 # حساب متوسط المكافأة بشكل آمن
                 avg_rwd_str = f"{current_avg_reward:.2f}" if np.isfinite(current_avg_reward) else "N/A"
                 print(f"Ep {ep+1}/{episodes} | Rwd: {total_reward:.2f} | AvgR: {avg_rwd_str} | Steps: {episode_steps} | Epsilon: {self.epsilon:.3f} | End: {reason}")

        print("Training finished.")
        plt.figure(figsize=(12, 6)); plt.plot(rewards_history, label='Episode Reward', alpha=0.7)
        if len(rewards_history) >= 50:
             valid_rewards = [r for r in rewards_history[-50:] if np.isfinite(r)] # تصفية القيم غير المحدودة
             if valid_rewards:
                 moving_avg = np.convolve(rewards_history, np.ones(50)/50, mode='valid') # حساب المتوسط المتحرك
                 plt.plot(np.arange(len(moving_avg)) + 49, moving_avg, label='50-ep MA', color='red')
        plt.title(f'IMRLS Training Performance'); plt.xlabel('Episode'); plt.ylabel('Total Reward')
        plt.legend(); plt.grid(True, linestyle=':'); plt.tight_layout()
        plt.savefig("imrls_training_performance.png"); plt.show(block=False) # عدم إيقاف التنفيذ

    def load_best_model(self):
        print(f"Attempting load: {self.model_save_path}, {self.arch_save_path}")
        if os.path.exists(self.model_save_path) and os.path.exists(self.arch_save_path):
            try:
                with open(self.arch_save_path, 'r') as f: arch_info = json.load(f)
                print(f"Loading architecture: {arch_info}")
                self.net = IntegratedEvolvingNetwork.from_architecture_info(arch_info).to(self.device)
                state_dict = torch.load(self.model_save_path, map_location=self.device)
                load_result = self.net.load_state_dict(state_dict, strict=False)
                print(f"State dict loaded. Missing: {load_result.missing_keys}, Unexpected: {load_result.unexpected_keys}")
                self.target_net = deepcopy(self.net).to(self.device); self.target_net.eval()
                current_lr = self.optimizer.defaults['lr'] if self.optimizer else 0.0005
                if not self._rebuild_optimizer(current_lr): print("Warn: Failed rebuild optimizer after loading.")
                print(f"Successfully loaded best model and architecture.")
            except Exception as e:
                print(f"Error loading model/arch: {e}. Re-initializing.")
                # استخدام الأبعاد الأولية المحفوظة لإعادة التهيئة
                self.net = IntegratedEvolvingNetwork(**self.initial_arch_info).to(self.device)
                self.target_net = deepcopy(self.net).to(self.device); self.target_net.eval()
                self._rebuild_optimizer(0.0005)
        else: print(f"No saved model/architecture found.")


# --- بيئة اختبار (يمكن استخدام بيئة Gym أو بيئة الطائرة المسيرة) ---
try: import gym
except ImportError: print("Warning: OpenAI Gym not installed."); gym = None

if gym:
    class GymEnvWrapper:
        def __init__(self, env_name='CartPole-v1'):
            try:
                self.env = gym.make(env_name)
                # فحص نوع فضاء الملاحظة
                if isinstance(self.env.observation_space, gym.spaces.Box):
                    self.state_dim = self.env.observation_space.shape[0]
                elif isinstance(self.env.observation_space, gym.spaces.Discrete):
                    self.state_dim = self.env.observation_space.n # قد لا يكون هذا مناسبًا كمدخل للشبكة مباشرة
                    print(f"Warning: Discrete observation space ({self.state_dim}). Encoding might be needed.")
                else:
                     raise NotImplementedError(f"Unsupported observation space type: {type(self.env.observation_space)}")

                 # فحص نوع فضاء الإجراء
                if isinstance(self.env.action_space, gym.spaces.Discrete):
                    self.action_dim = self.env.action_space.n
                elif isinstance(self.env.action_space, gym.spaces.Box):
                    self.action_dim = self.env.action_space.shape[0]
                    print(f"Warning: Continuous action space ({self.action_dim}). IMRLS expects discrete actions.")
                else:
                    raise NotImplementedError(f"Unsupported action space type: {type(self.env.action_space)}")

                print(f"Gym Env: {env_name}, State Dim: {self.state_dim}, Action Dim: {self.action_dim}")
            except Exception as e:
                print(f"Error initializing Gym environment '{env_name}': {e}")
                self.env = None # تعيين None للإشارة للفشل

        def reset(self):
            if self.env is None: return None
            try:
                state = self.env.reset()
                return state[0] if isinstance(state, tuple) else state
            except Exception as e:
                print(f"Error during env.reset(): {e}")
                return None

        def step(self, action):
            if self.env is None: return None, 0, True, {} # إرجاع حالة انتهاء إذا فشلت التهيئة
            try:
                if isinstance(action, torch.Tensor): action = action.item()
                result = self.env.step(action)
                if len(result) == 5: next_state, reward, terminated, truncated, info = result; done = terminated or truncated
                elif len(result) == 4: next_state, reward, done, info = result
                else: raise ValueError(f"Unexpected output from env.step: {result}")
                if isinstance(next_state, tuple): next_state = next_state[0]
                return next_state, reward, done, info
            except Exception as e:
                print(f"Error during env.step(): {e}")
                return None, 0, True, {'error': str(e)} # إرجاع حالة انتهاء مع خطأ

        def close(self):
             if self.env: self.env.close()

# --- التشغيل الرئيسي ---
if __name__ == "__main__" and gym:
    print("\n--- Starting IMRLS Example on CartPole-v1 ---")
    seed = 44; random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

    env_wrapper = GymEnvWrapper(env_name='CartPole-v1')
    # التأكد من تهيئة البيئة بنجاح
    if env_wrapper.env is None:
        print("Failed to initialize environment. Exiting.")
        exit()
    # التأكد من أن فضاء الإجراء منفصل
    if not isinstance(env_wrapper.env.action_space, gym.spaces.Discrete):
        print(f"Error: IMRLS requires a discrete action space, but got {type(env_wrapper.env.action_space)}. Exiting.")
        exit()

    state_dim = env_wrapper.state_dim; action_dim = env_wrapper.action_dim

    # --- استخدام AdamW وبنية أولية بسيطة ---
    agent = IMRLS_Trainer(input_dim=state_dim, action_dim=action_dim,
                          hidden_dims=[64], use_dynamic_units=False, use_chaos_optimizer=False,
                          learning_rate=0.001, tau_update=False, buffer_size=20000)

    agent.train(env_wrapper, episodes=250, max_steps_per_episode=500, save_best_model=True)

    print("\n--- Testing Trained IMRLS Agent ---")
    agent.load_best_model()
    test_episodes = 5; total_rewards_test = []
    for ep in range(test_episodes):
        state = env_wrapper.reset()
        if state is None: print("Failed to reset env for test."); break # تحقق من إعادة التعيين
        if not isinstance(state, np.ndarray): state = np.array(state, dtype=np.float32)
        done = False; episode_reward = 0; steps = 0
        while not done and steps < 500:
            action = agent.select_action(state) # Epsilon منخفض
            next_state, reward, done, info = env_wrapper.step(action)
            if next_state is None: print("Error during test step. Ending episode."); break # تحقق من الخطوة
            if not isinstance(next_state, np.ndarray): next_state = np.array(next_state, dtype=np.float32)
            state = next_state; episode_reward += reward; steps += 1
        total_rewards_test.append(episode_reward)
        print(f"Test Episode {ep+1}/{test_episodes} | Reward: {episode_reward} | Steps: {steps}")
    if total_rewards_test:
        print(f"\nAverage Test Reward: {np.mean(total_rewards_test):.2f} +/- {np.std(total_rewards_test):.2f}")
    env_wrapper.close()

    '''
    الكود المقدم يُظهر نظامًا متكاملًا للتعلم المعزز الرياضي (IMRLS - Integrated Mathematical Reinforcement Learning System ) يدمج بين مفاهيم الشبكات العصبية الديناميكية القابلة للتطور وميكانيكيات التعلم المعزز المُحسَّنة. إليك التفاصيل:

الهدف الرئيسي:
بناء نظام وكيل ذكاء اصطناعي قادر على:

التعلم في بيئات معقدة (مثل بيئة CartPole-v1 من OpenAI Gym).
التطور التلقائي للبنية أثناء التدريب بإضافة طبقات جديدة عند استشعار توقف التحسن.
التكامل بين التعلم المعزز (Q-Learning) والشبكات العصبية الديناميكية (DynamicMathUnit, TauRLayer).
المكونات الرئيسية:
DynamicMathUnit :
وحدة رياضية تدمج دوالًا غير خطية (مثل sin, cos, SiLU) مع معاملات قابلة للتعلم.
تُحسّن تمثيل العلاقات الديناميكية عبر تحويلات مُعقدة (أسس، تركيب دالي).
TauRLayer :
طبقة مخصصة تحسب قيمة Tau لموازنة "التقدم" و"المخاطرة" في عملية التعلم.
تستخدم tanh و clamp لضمان استقرار القيم.
IntegratedEvolvingNetwork :
شبكة عصبية تتطور ذاتيًا بإضافة طبقات جديدة (add_layer) عند توقف تحسن الأداء.
تدمج بين:
وحدات DynamicMathUnit (اختياري).
طبقات TauRLayer لتحسين قيم Q.
آلية evolve_structure لاتخاذ قرار التطور بناءً على سجل الخسائر.
IMRLS_Trainer :
نظام التدريب الرئيسي، يدعم:
ذاكرة التجارب (Replay Buffer) لتخزين الخبرات.
الشبكة الهدف (Target Network) لتحديث مستقر.
استكشاف متكيف (Epsilon-Greedy) مع تناقص ε تدريجي.
دعم بيئة Gym (مثل CartPole-v1).
ChaosOptimizer (اختياري):
مُحسِّن مُخصص يعتمد على نظام لورينتز الفوضوي لتحديث الأوزان.
خطوات العمل:
التهيئة :
تهيئة الشبكة (IntegratedEvolvingNetwork) مع بنية أولية بسيطة.
ضبط المعلمات مثل معدل التعلم (learning_rate=0.001) وحجم الذاكرة (buffer_size=20000).
التدريب :
في كل حلقة تدريب:
يختار الوكيل إجراءً (بناءً على ε-Greedy).
يخزن التجربة (الحالة، الإجراء، المكافأة، الحالة التالية) في الذاكرة.
يُحدّث الشبكة باستخدام عينات عشوائية من الذاكرة.
يُطوّر الشكل الهندسي للشبكة (add_layer) إذا لزم الأمر.
التحديثات :
تحديث الشبكة الهدف كل update_target_every خطوة.
استخدام AdamW أو ChaosOptimizer حسب الاختيار.
تقييد التدرجات (clip_grad_norm_) لمنع الانفجار.
الاختبار :
بعد التدريب، يُختبر الأداء باستخدام النموذج الأفضل (المُخزَّن) مع ε منخفض.
التحسينات الرئيسية:
التكامل مع بيئة Gym : دعم بيئات مثل CartPole-v1 عبر GymEnvWrapper.
التطور الهيكلي : إضافة طبقات جديدة (nn.Sequential) عند استشعار توقف التحسن.
استخدام TauRLayer : لتحسين قيم Q عبر دمج مفاهيم "التقدم" و"المخاطرة".
دعم الحفظ/التحميل : حفظ أفضل نموذج وهندسة (best_imrls_model.pth).
الناتج المتوقع:
رسم بياني لأداء التدريب مع متوسط متحرك (50 حلقة).
طباعة تفاصيل الأداء (المكافأة، عدد الخطوات، قيمة ε) كل 20 حلقة.
نتائج الاختبار بعد التدريب (متوسط المكافأة في 5 حلقات).
الخلاصة:
الكود يُظهر نظامًا متطورًا للتعلم المعزز يدمج بين الشبكات العصبية الديناميكية وميكانيكيات الاستقرار الرياضي، مما يجعله مناسبًا للبيئات المعقدة التي تتطلب تكيفًا مستمرًا في الهيكل.
    '''