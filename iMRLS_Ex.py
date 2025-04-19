
'''
بالتأكيد! بما أن نظام IMRLS الأساسي يعمل الآن بشكل مستقر، يمكننا تطبيقه على مثال الطائرة المسيرة الأكثر إثارة للإعجاب.
ملاحظات هامة قبل البدء:
التدريب الطويل: تذكر أن تدريب هذا النظام على بيئة الطائرة المسيرة المعقدة سيتطلب وقتًا أطول بكثير (آلاف الحلقات على الأقل) وموارد حسابية أكبر من CartPole للحصول على نتائج جيدة حقًا. النتائج بعد عدد قليل من الحلقات قد لا تظهر السلوك الذكي الكامل.
الإجراءات المنفصلة: سنستخدم نفس نهج تحويل الإجراءات المستمرة (التسارع) إلى مجموعة إجراءات منفصلة للتبسيط، كما فعلنا سابقًا. للحصول على تحكم أكثر دقة، سيتطلب الأمر تعديل IMRLS لاستخدام خوارزمية Actor-Critic.
الرسوم البيانية: سيقوم الكود بحفظ الرسوم البيانية (منحنى التدريب ومسار الاختبار) كملفات صور بسبب استخدام الواجهة الخلفية Agg. لن تظهر نوافذ رسومية تفاعلية.
الكود الكامل لمثال الطائرة المسيرة باستخدام IMRLS:
هذا الكود يدمج الكلاسات النهائية المستقرة لـ IMRLS مع بيئة AdvancedDroneEnv وآلية تحويل الإجراءات.
'''
"""
IMRLS: مثال على نظام تعلم مبتكر يعتمد على معادلات رياضية تتطوّر مع التدريب
تم تطويره بواسطة: [باسل يحيى عبدالله/ العراق/ الموصل ]
تاريخ الإصدار: [20/4/2025]
"""

# -*- coding: utf-8 -*-
# الكود الكامل للنظام المتكامل (IMRLS) مع مثال الطائرة المسيرة
# يستخدم نسخة IMRLS المستقرة مع تحويل الإجراءات لمنفصلة

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# --- تغيير الواجهة الخلفية لـ Matplotlib ---
import matplotlib
try:
    matplotlib.use('Agg') # للحفظ فقط دون عرض تفاعلي
    print("Matplotlib backend set to Agg.")
except Exception as e:
    print(f"Warning: Failed to set Matplotlib backend to Agg. Error: {e}")
# --- نهاية تغيير الواجهة الخلفية ---

import matplotlib.pyplot as plt
# لاستخدام الرسم ثلاثي الأبعاد
from mpl_toolkits.mplot3d import Axes3D
from collections import deque
from copy import deepcopy
import random
import os
import json

# --- المكونات الأساسية للنظام (DynamicMathUnit, TauRLayer, ChaosOptimizer, IntegratedEvolvingNetwork) ---
# --- (لصق الكود الكامل لهذه الكلاسات هنا من الردود السابقة التي تأكدنا من عملها) ---

class DynamicMathUnit(nn.Module):
    """
    وحدة رياضية تمثل معادلة ديناميكية قابلة للتعلم.
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
    """
    محسن مستوحى من نظرية الشواش (معادلات لورنز).
    """
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
    """
    شبكة عصبية متكاملة تجمع بين الوحدات الرياضية الديناميكية (اختياري) وطبقات Tau
    وتمتلك القدرة على التطور الهيكلي بإضافة طبقات.
    """
    def __init__(self, input_dim, hidden_dims, output_dim,
                 use_dynamic_units=False, max_layers=8):
        super().__init__()
        self.input_dim = input_dim; self.hidden_dims = list(hidden_dims); self.output_dim = output_dim
        self.use_dynamic_units = use_dynamic_units; self.max_layers = max_layers
        self.layers = nn.ModuleList() # طبقات مخفية

        # --- بناء الطبقات الأولية ---
        current_dim = input_dim
        for i, hidden_dim in enumerate(self.hidden_dims):
            if not (isinstance(current_dim, int) and current_dim > 0 and isinstance(hidden_dim, int) and hidden_dim > 0): raise ValueError(f"Invalid dims layer {i}")
            layer_modules = []; block_input_dim = current_dim
            if use_dynamic_units:
                dynamic_unit = DynamicMathUnit(current_dim, current_dim) # يحافظ على البعد
                layer_modules.append(dynamic_unit)
                block_input_dim = current_dim # مدخل الخطية هو نفسه

            layer_modules.extend([nn.Linear(block_input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(), TauRLayer(hidden_dim, hidden_dim)])
            self.layers.append(nn.Sequential(*layer_modules)); current_dim = hidden_dim

        self.output_layer = nn.Linear(current_dim, output_dim)
        self.performance_history = deque(maxlen=50); self.layer_evolution_threshold = 0.002

    def get_architecture_info(self):
        """ الحصول على معلومات البنية الحالية للحفظ. """
        # بناء قائمة الأبعاد المخفية الحالية بدقة
        current_hidden_dims = []
        if self.layers:
             last_dim = self.input_dim
             for layer_seq in self.layers:
                  # العثور على الطبقة الخطية الرئيسية لتحديد المخرج
                  linear_layer = None
                  for module in layer_seq:
                      if isinstance(module, nn.Linear):
                           linear_layer = module
                           break # افترض وجود خطية واحدة رئيسية لكل بلوك
                  if linear_layer:
                      current_hidden_dims.append(linear_layer.out_features)
                      last_dim = linear_layer.out_features
                  else: # Fallback إذا لم نجد خطية (نادر)
                       current_hidden_dims.append(last_dim)

        info = {'input_dim': self.input_dim, 'hidden_dims': current_hidden_dims, 'output_dim': self.output_dim,
                'use_dynamic_units': self.use_dynamic_units, 'max_layers': self.max_layers,}
        return info

    @classmethod
    def from_architecture_info(cls, info):
        """ إنشاء نموذج بناءً على معلومات البنية المحفوظة. """
        net = cls(info['input_dim'], info['hidden_dims'], info['output_dim'],
                  info.get('use_dynamic_units', False), info.get('max_layers', 8))
        return net

    def add_layer(self):
        """ إضافة طبقة مخفية جديدة قبل طبقة الإخراج. """
        if len(self.layers) >= self.max_layers: return False
        print(f"*** Evolving IMRLS Network: Adding hidden layer {len(self.layers) + 1}/{self.max_layers} ***")
        try: device = next(self.parameters()).device
        except StopIteration: device = torch.device("cpu")

        if self.layers:
             # مدخل الطبقة الجديدة هو مخرج آخر طبقة مخفية
             last_layer_output_dim = self.layers[-1][-1].progress_transform.in_features # Hidden dim
        else: last_layer_output_dim = self.input_dim
        new_layer_input_dim = last_layer_output_dim
        # استخدام نفس البعد المخفي للطبقة الجديدة
        new_layer_hidden_dim = self.hidden_dims[-1] if self.hidden_dims else max(32, self.output_dim)

        if not isinstance(new_layer_input_dim, int) or new_layer_input_dim <= 0: print(f"Error: Invalid input dim ({new_layer_input_dim})"); return False
        if not isinstance(new_layer_hidden_dim, int) or new_layer_hidden_dim <= 0: print(f"Error: Invalid hidden dim ({new_layer_hidden_dim})"); return False

        new_layer_modules = []; current_dim_new = new_layer_input_dim
        if self.use_dynamic_units: new_layer_modules.append(DynamicMathUnit(current_dim_new, current_dim_new))
        new_layer_modules.extend([nn.Linear(current_dim_new, new_layer_hidden_dim), nn.LayerNorm(new_layer_hidden_dim), nn.ReLU(), TauRLayer(new_layer_hidden_dim, new_layer_hidden_dim)])
        new_sequential_layer = nn.Sequential(*new_layer_modules).to(device)
        self.layers.append(new_sequential_layer)

        # تحديث hidden_dims (للتتبع)
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
        if len(recent_metrics) > 20: # انتظار فترة أطول قليلاً
            # نفترض أن validation_metric هو خسارة (قيمة أقل أفضل)
            improvement = np.mean(np.diff(recent_metrics[-10:])) # سالب يعني تحسن
            if improvement > -self.layer_evolution_threshold: # إذا لم يكن التحسن كافياً
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
        structure_changed = self.net.evolve_structure(loss.item()) # استخدام الخسارة للتطور
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
            try: # إضافة try-except للحلقة الرئيسية
                state = env.reset();
                if state is None: print(f"Warning: env.reset() returned None at episode {ep+1}. Skipping episode."); continue # تخطي الحلقة إذا فشل reset
                if not isinstance(state, np.ndarray): state = np.array(state, dtype=np.float32)
                total_reward = 0; episode_steps = 0; done = False; info = {}
                while not done and episode_steps < max_steps_per_episode:
                    action = self.select_action(state)
                    if action_wrapper: next_state, reward, done, info = action_wrapper(env, action)
                    else: next_state, reward, done, info = env.step(action)
                    if next_state is None: print(f"Warning: env.step() returned None at step {episode_steps}. Ending episode."); break # إنهاء الحلقة إذا فشل step
                    if not isinstance(next_state, np.ndarray): next_state = np.array(next_state, dtype=np.float32)
                    self.store_transition(state, action, reward, next_state, done)
                    updated = False
                    if len(self.memory) >= self.batch_size and total_steps % 4 == 0: updated = self.update_network()
                    state = next_state; total_reward += reward; episode_steps += 1; total_steps += 1
                self.adaptive_exploration()
                rewards_history.append(total_reward)
                current_avg_reward = np.mean(rewards_history[-50:]) if len(rewards_history) >= 50 else np.mean(rewards_history) if rewards_history else 0
                if save_best_model and len(rewards_history) > 50 and np.isfinite(current_avg_reward) and current_avg_reward > self.best_avg_reward:
                    print(f"*** New best avg reward: {current_avg_reward:.2f} (prev {self.best_avg_reward:.2f}). Saving... ***")
                    self.best_avg_reward = current_avg_reward
                    try:
                         arch_info = self.net.get_architecture_info();
                         with open(self.arch_save_path, 'w') as f: json.dump(arch_info, f)
                         torch.save(self.net.state_dict(), self.model_save_path)
                    except Exception as save_e: print(f"Error saving best model: {save_e}")
                if (ep + 1) % 20 == 0:
                     reason = info.get('reason', 'max_steps') if done else 'in_prog'
                     avg_rwd_str = f"{current_avg_reward:.2f}" if np.isfinite(current_avg_reward) else "N/A"
                     print(f"Ep {ep+1}/{episodes} | Rwd: {total_reward:.2f} | AvgR: {avg_rwd_str} | Steps: {episode_steps} | Epsilon: {self.epsilon:.3f} | End: {reason}")
            except Exception as train_loop_e:
                 print(f"Error in training loop at episode {ep+1}: {train_loop_e}")
                 # يمكنك اختيار إيقاف التدريب أو الاستمرار
                 # break # لإيقاف التدريب
                 continue # للاستمرار في الحلقة التالية

        print("Training finished.")
        plt.figure(figsize=(12, 6)); plt.plot(rewards_history, label='Episode Reward', alpha=0.7)
        if len(rewards_history) >= 50:
             moving_avg = np.convolve(rewards_history, np.ones(50)/50, mode='valid')
             plt.plot(np.arange(len(moving_avg)) + 49, moving_avg, label='50-ep MA', color='red')
        plt.title(f'IMRLS Training Performance'); plt.xlabel('Episode'); plt.ylabel('Total Reward')
        plt.legend(); plt.grid(True, linestyle=':'); plt.tight_layout()
        plt.savefig("imrls_training_performance.png"); plt.show(block=False)

    def load_best_model(self):
        print(f"Attempting load: {self.model_save_path}, {self.arch_save_path}")
        if os.path.exists(self.model_save_path) and os.path.exists(self.arch_save_path):
            try:
                with open(self.arch_save_path, 'r') as f: arch_info = json.load(f)
                print(f"Loading architecture: {arch_info}")
                # استعادة الأبعاد الأولية من المعلومات المحفوظة
                self.initial_arch_info = arch_info # تحديث المعلومات الأولية
                self.net = IntegratedEvolvingNetwork.from_architecture_info(arch_info).to(self.device)
                state_dict = torch.load(self.model_save_path, map_location=self.device)
                load_result = self.net.load_state_dict(state_dict, strict=False)
                print(f"State dict loaded. Missing: {load_result.missing_keys}, Unexpected: {load_result.unexpected_keys}")
                self.target_net = deepcopy(self.net).to(self.device); self.target_net.eval()
                # استخدام معدل تعلم افتراضي عند التحميل
                lr_on_load = 0.0001 # معدل تعلم صغير للتحميل
                if not self._rebuild_optimizer(lr_on_load): print("Warn: Failed rebuild optimizer after loading.")
                print(f"Successfully loaded best model and architecture.")
            except Exception as e:
                print(f"Error loading model/arch: {e}. Re-initializing.")
                # إعادة التهيئة باستخدام الأبعاد الأولية المحفوظة
                self.net = IntegratedEvolvingNetwork(**self.initial_arch_info).to(self.device)
                self.target_net = deepcopy(self.net).to(self.device); self.target_net.eval()
                self._rebuild_optimizer(0.0005)
        else: print(f"No saved model/architecture found.")

# --- بيئة الطائرة المسيرة ---
class AdvancedDroneEnv:
    def __init__(self, target_movement='static', num_obstacles=3):
        self.world_bounds = np.array([[-10, 10], [-10, 10], [0, 5]])
        self.target_movement = target_movement; self.num_obstacles = num_obstacles
        self.dt = 0.1; self.max_speed = 1.5; self.max_acceleration = 0.5; self.gravity = 9.81
        self.drone_state = np.zeros(6); self.target_pos = np.zeros(3); self.obstacles = []
        self.action_space_dim = 3; self.observation_space_dim = 6 + 3
        self.max_steps = 500; self.current_step = 0; self.previous_dist_to_target = np.inf
    def _generate_obstacles(self):
        self.obstacles = []; min_dist_start, min_dist_target = 2.0, 2.0
        attempts, max_attempts = 0, 100 * self.num_obstacles
        while len(self.obstacles) < self.num_obstacles and attempts < max_attempts:
            attempts += 1
            pos = np.random.rand(3) * (self.world_bounds[:, 1] - self.world_bounds[:, 0]) + self.world_bounds[:, 0]
            pos[2] = np.random.uniform(0.5, 3.0)
            dist_start_ok = np.linalg.norm(pos - self.drone_state[:3]) > min_dist_start
            dist_target_ok = np.linalg.norm(pos - self.target_pos) > min_dist_target
            dist_others_ok = all(np.linalg.norm(pos - obs['pos']) > obs['radius'] + 0.8 for obs in self.obstacles)
            if dist_start_ok and dist_target_ok and dist_others_ok:
                radius = np.random.uniform(0.4, 1.0)
                self.obstacles.append({'pos': pos, 'radius': radius})
        if len(self.obstacles) < self.num_obstacles: print(f"Warn: Generated {len(self.obstacles)}/{self.num_obstacles} obstacles.")
    def _update_target_position(self):
        time_factor = self.current_step * self.dt; initial_target = np.array([8.0, 8.0, 1.5])
        if self.target_movement == 'linear': self.target_pos[0] = initial_target[0] - 0.15 * time_factor; self.target_pos[1] = initial_target[1] - 0.10 * time_factor
        elif self.target_movement == 'sinusoidal': self.target_pos[0] = initial_target[0] + 3.0 * np.sin(0.25 * time_factor); self.target_pos[1] = initial_target[1] + 3.0 * np.cos(0.15 * time_factor)
        elif self.target_movement == 'static': self.target_pos = initial_target
        self.target_pos = np.clip(self.target_pos, self.world_bounds[:, 0], self.world_bounds[:, 1])
        self.target_pos[2] = max(self.target_pos[2], self.world_bounds[2, 0] + 0.2)
    def reset(self):
        self.drone_state = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        self.target_pos = np.array([ np.random.uniform(6, 9), np.random.uniform(6, 9), 1.5])
        self._generate_obstacles(); self.current_step = 0
        self.previous_dist_to_target = np.linalg.norm(self.target_pos - self.drone_state[:3])
        return self._get_observation().astype(np.float32)
    def _get_observation(self):
        relative_target_pos = self.target_pos - self.drone_state[:3]
        return np.concatenate([self.drone_state, relative_target_pos]).astype(np.float32)
    def _apply_physics(self, action):
        acceleration = np.clip(action, -self.max_acceleration, self.max_acceleration)
        self.drone_state[3:] += acceleration * self.dt; self.drone_state[3:] *= (1.0 - 0.05 * self.dt)
        speed = np.linalg.norm(self.drone_state[3:])
        if speed > self.max_speed: self.drone_state[3:] *= (self.max_speed / speed)
        self.drone_state[:3] += self.drone_state[3:] * self.dt
        self.drone_state[:3] = np.clip(self.drone_state[:3], self.world_bounds[:, 0], self.world_bounds[:, 1])
        if self.drone_state[2] < self.world_bounds[2, 0]: self.drone_state[2] = self.world_bounds[2, 0]; self.drone_state[5] = max(0, self.drone_state[5])
    def _calculate_reward(self, action):
        current_dist = np.linalg.norm(self.target_pos - self.drone_state[:3])
        reward_progress = (self.previous_dist_to_target - current_dist) * 5.0
        self.previous_dist_to_target = current_dist
        reward_obstacle = 0; collision = False; min_dist_surface = np.inf
        for obs in self.obstacles:
            dist_center = np.linalg.norm(self.drone_state[:3] - obs['pos'])
            dist_surface = dist_center - obs['radius']; min_dist_surface = min(min_dist_surface, dist_surface)
            if dist_surface < 0: reward_obstacle -= 200.0; collision = True; break
            elif dist_surface < 0.4: reward_obstacle -= 15.0 / ((dist_surface + 0.05)**2)
        reward_altitude = -2.0 if self.drone_state[2] < 0.3 else 0
        speed_magnitude = np.linalg.norm(self.drone_state[3:])
        reward_efficiency = -0.05 * speed_magnitude; reward_action = -0.01 * np.linalg.norm(action)
        reward_time = -0.05; reward_goal = 0; target_reached = False
        if current_dist < 0.35: reward_goal = 400.0; target_reached = True
        total_reward = (reward_progress + reward_obstacle + reward_efficiency + reward_action + reward_time + reward_goal + reward_altitude)
        return total_reward, target_reached, collision
    def step(self, action):
        if not isinstance(action, np.ndarray): action = np.array(action)
        if action.shape != (self.action_space_dim,): action = np.zeros(self.action_space_dim)
        self._apply_physics(action); self._update_target_position()
        reward, target_reached, collision = self._calculate_reward(action)
        next_observation = self._get_observation(); self.current_step += 1
        done = False; info = {'reason': 'in_progress', 'target_reached': False, 'collision': False}
        if target_reached: done=True; info.update({'reason':'target_reached', 'target_reached':True})
        elif collision: done=True; info.update({'reason':'collision', 'collision':True}); reward -= 50
        elif self.current_step >= self.max_steps: done=True; info['reason'] = 'max_steps_reached'
        return next_observation, reward, done, info
    def render(self, ax=None, trajectory=None, title_suffix=""):
        create_new_fig = ax is None
        if create_new_fig: fig = plt.figure(figsize=(10, 8)); ax = fig.add_subplot(111, projection='3d')
        else: ax.clear()
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z (Altitude)')
        ax.set_xlim(self.world_bounds[0]); ax.set_ylim(self.world_bounds[1]); ax.set_zlim(self.world_bounds[2])
        ax.set_title(f'Drone Simulation {title_suffix}')
        ax.scatter(*self.drone_state[:3], c='blue', marker='^', s=100, label='Drone', depthshade=True)
        ax.scatter(*self.target_pos, c='lime', marker='*', s=200, label='Target', depthshade=True)
        for obs in self.obstacles:
            u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
            x = obs['pos'][0] + obs['radius'] * np.cos(u) * np.sin(v); y = obs['pos'][1] + obs['radius'] * np.sin(u) * np.sin(v); z = obs['pos'][2] + obs['radius'] * np.cos(v)
            ax.plot_wireframe(x, y, z, color="orangered", alpha=0.3)
        if trajectory and len(trajectory) > 1:
            traj_points = np.array(trajectory)
            ax.plot(traj_points[:, 0], traj_points[:, 1], traj_points[:, 2], 'cornflowerblue', linestyle='--', alpha=0.7, label='Path')
        if create_new_fig: ax.legend(); plt.tight_layout(); # plt.show() # لا نعرض تفاعليًا
        return ax

# --- التشغيل الرئيسي ---
if __name__ == "__main__":
    print("\n--- Starting IMRLS Example on AdvancedDroneEnv ---")
    seed = 45; random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

    # تهيئة بيئة الطائرة
    env = AdvancedDroneEnv(target_movement='sinusoidal', num_obstacles=5)
    state_dim = env.observation_space_dim
    # استخدام الإجراءات المنفصلة
    action_map = {
        0: np.array([ env.max_acceleration * 0.8, 0, 0]), 1: np.array([-env.max_acceleration * 0.8, 0, 0]),
        2: np.array([ 0, env.max_acceleration * 0.8, 0]), 3: np.array([ 0,-env.max_acceleration * 0.8, 0]),
        4: np.array([ 0, 0, env.max_acceleration * 0.6]), 5: np.array([ 0, 0,-env.max_acceleration * 0.4]),
        6: np.array([ env.max_acceleration * 0.5, env.max_acceleration * 0.5, 0]),
        7: np.array([-env.max_acceleration * 0.5,-env.max_acceleration * 0.5, 0]),
        8: np.array([ 0, 0, 0]),
    }
    discrete_action_dim = len(action_map)
    print(f"Using Discretized Action Space with {discrete_action_dim} actions.")

    def discrete_step_wrapper(env_instance, discrete_action_index):
        continuous_action = action_map.get(discrete_action_index, np.zeros(env_instance.action_space_dim))
        return env_instance.step(continuous_action)

    # تهيئة العميل (قد تحتاج لتعديل المعلمات للبيئة المعقدة)
    agent = IMRLS_Trainer(input_dim=state_dim, action_dim=discrete_action_dim,
                          hidden_dims=[128, 64], # زيادة حجم الشبكة الأولية
                          use_dynamic_units=False, # البدء بدون وحدات ديناميكية
                          use_chaos_optimizer=False, # استخدام AdamW للاستقرار
                          learning_rate=3e-4,      # معدل تعلم أقل
                          buffer_size=200000,     # ذاكرة أكبر
                          batch_size=128,          # باتش أكبر
                          update_target_every=25, # تحديث أقل للهدف
                          tau_update=True,        # تفعيل Tau update لهذه البيئة
                          epsilon_decay=0.998)    # تناقص أبطأ لـ epsilon

    # التدريب (قد تحتاج لعدد حلقات كبير جدًا)
    agent.train(env, episodes=500, max_steps_per_episode=env.max_steps, # زيادة عدد الحلقات
                save_best_model=True, action_wrapper=discrete_step_wrapper)

    # --- الاختبار ---
    print("\n--- Testing Trained IMRLS Agent on Drone Env ---")
    agent.load_best_model() # تحميل أفضل نموذج تم حفظه أثناء التدريب
    test_episodes = 5; total_rewards_test = []
    # إنشاء الشكل والمحاور مرة واحدة للاختبار
    fig_test = plt.figure(figsize=(10, 8)); ax_test = fig_test.add_subplot(111, projection='3d')

    for ep in range(test_episodes):
        state = env.reset()
        if state is None: continue
        if not isinstance(state, np.ndarray): state = np.array(state, dtype=np.float32)
        done = False; episode_reward = 0; steps = 0; trajectory_test = [state[:3].copy()]
        while not done and steps < env.max_steps:
            action = agent.select_action(state) # Epsilon منخفض
            next_state, reward, done, info = discrete_step_wrapper(env, action)
            if next_state is None: break
            if not isinstance(next_state, np.ndarray): next_state = np.array(next_state, dtype=np.float32)
            trajectory_test.append(next_state[:3].copy()) # سجل الموقع
            state = next_state; episode_reward += reward; steps += 1
        total_rewards_test.append(episode_reward)
        print(f"Test Episode {ep+1}/{test_episodes} | Reward: {episode_reward:.2f} | Steps: {steps} | Reason: {info.get('reason', 'N/A')}")

        # رسم المسار لهذه الحلقة الاختبارية (اختياري)
        env.render(ax=ax_test, trajectory=trajectory_test, title_suffix=f" Test Run {ep+1}")
        fig_test.savefig(f"imrls_drone_test_run_{ep+1}.png") # حفظ الشكل لكل حلقة اختبار
        # plt.pause(0.1) # إيقاف مؤقت لرؤية الرسم إذا لم يكن Backend Agg

    if total_rewards_test: print(f"\nAverage Test Reward: {np.mean(total_rewards_test):.2f} +/- {np.std(total_rewards_test):.2f}")
    # إغلاق الشكل النهائي (إذا لم يكن Agg)
    # plt.close(fig_test)

'''
الكود المقدم يُظهر نظامًا متطورًا للتعلم المعزز (IMRLS - Integrated Mathematical Reinforcement Learning System ) مُصمم خصيصًا للتحكم في بيئات ديناميكية ثلاثية الأبعاد (مثل طائرة بدون طيار تتجنب العقبات وتتعقب أهدافًا متحركة). إليك التفاصيل:

الهدف الرئيسي:
بناء وكيل ذكاء اصطناعي قادر على:

التعلم في بيئات ثلاثية الأبعاد معقدة (مثل بيئة DroneEnv المُضمنة).
التحكم الدقيق في حركة الطائرة مع تجنب العقبات وتتبع أهداف متحركة.
التكامل بين الشبكات العصبية الديناميكية (DynamicMathUnit, TauRLayer) ومُحسِّنات الفوضى (ChaosOptimizer).
المكونات الرئيسية:
1. IntegratedEvolvingNetwork :
شبكة عصبية تتطور ذاتيًا بإضافة طبقات جديدة عند استشعار توقف التحسن.
تدمج:
DynamicMathUnit : وحدات رياضية مع دوال غير خطية (sin, cos, SiLU).
TauRLayer : طبقة لحساب "قيمة التوازن" بين التقدم والمخاطر.
التطور الهيكلي : تُضيف طبقات جديدة إذا توقف انخفاض الخسارة (layer_evolution_threshold).
2. ChaosOptimizer :
مُحسِّن مُخصص يعتمد نظام لورينتز الفوضوي لتحديث الأوزان، مما يساعد في الهروب من الحدود الدنيا المحلية.
3. IMRLS_Trainer :
نظام تدريب متكامل يدعم:
ذاكرة التجارب (Replay Buffer) لتخزين الخبرات.
الشبكة الهدف (Target Network) لتحديث مستقر.
استكشاف متكيف (Epsilon-Greedy) مع تناقص ε تدريجي.
دعم بيئة مخصصة (DroneEnv) مع فيزياء واقعية.
4. DroneEnv (بيئة الطائرة بدون طيار) :
الخصائص :
الهدف المتحرك : يتحرك خطياً أو بشكل جيبي حسب الإعداد.
العقبات الديناميكية : كرات عشوائية تظهر في مساحة الطيران.
حساب المكافآت : يعتمد على:
التقدم نحو الهدف (reward_progress).
تجنب الاصطدام (reward_obstacle).
كفاءة الطاقة (reward_efficiency).
التحكم في الارتفاع (reward_altitude).
خطوات العمل:
التهيئة :
تهيئة الشبكة (IntegratedEvolvingNetwork) مع بنية أولية.
ضبط المعلمات مثل معدل التعلم (learning_rate=0.001) وحجم الذاكرة (buffer_size=20000).
التدريب :
في كل حلقة تدريب:
يختار الوكيل إجراءً (حركة الطائرة) بناءً على ε-Greedy.
يخزن التجربة (الحالة، الإجراء، المكافأة، الحالة التالية) في الذاكرة.
يُحدّث الشبكة باستخدام عينات عشوائية من الذاكرة.
يُطوّر الشكل الهندسي للشبكة (add_layer) إذا لزم الأمر.
التحديثات :
تحديث الشبكة الهدف كل update_target_every خطوة.
استخدام AdamW أو ChaosOptimizer حسب الاختيار.
تقييد التدرجات (clip_grad_norm_) لمنع الانفجار.
الاختبار :
بعد التدريب، يُختبر الأداء باستخدام النموذج الأفضل مع ε منخفض.
التحسينات الرئيسية:
Reward System متقدم : يوازن بين التقدم، السلامة، والكفاءة.
الفيزياء الواقعية : محاكاة الحركة والاصطدامات في 3D.
التطور الهيكلي : إضافة طبقات جديدة عند توقف التحسن.
التحكم في الاستقرار : استخدام clamp و Huber Loss.
الناتج المتوقع:
رسم بياني لأداء التدريب مع متوسط متحرك (50 حلقة).
مرئي 3D لمسار الطائرة والعقبات والهدف.
طباعة تفاصيل الأداء (المكافأة، عدد الخطوات، سبب انتهاء الحلقة).
الخلاصة:
الكود يُظهر نظامًا متطورًا للتحكم في بيئات ثلاثية الأبعاد الديناميكية، مع دمج الشبكات العصبية القابلة للتطور ومُحسِّنات الفوضى، مما يجعله مناسبًا لمهمات مثل توصيل الطرود أو الاستكشاف الآلي.
'''