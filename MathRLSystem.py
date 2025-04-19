# -*- coding: utf-8 -*-
# الكود الكامل لنظام التعلم المعزز الرياضي (MathRLSystem)
# نسخة محسنة للاستقرار (استخدام AdamW، tanh في TauLayer، ...)

"""
: نظام تعلم معزز مبتكر يعتمد على معادلات رياضية تتطوّر مع التدريب
تم تطويره بواسطة: [باسل يحيى عبدالله/ العراق/ الموصل ]
تاريخ الإصدار: [20/4/2025]
"""

import torch
import torch.nn as nn
import torch.optim as optim # <-- استيراد optim لاستخدام AdamW
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt
from copy import deepcopy

# --- المكونات الأساسية ---

class DynamicMathUnit(nn.Module):
    def __init__(self, input_dim, output_dim, complexity=5):
        super().__init__()
        if not (isinstance(input_dim, int) and input_dim > 0 and
                isinstance(output_dim, int) and output_dim > 0 and
                isinstance(complexity, int) and complexity > 0):
            raise ValueError("input_dim, output_dim, and complexity must be positive integers.")
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.complexity = complexity
        self.internal_dim = max(output_dim, complexity, input_dim // 2 + 1)
        self.input_layer = nn.Linear(input_dim, self.internal_dim)
        self.output_layer = nn.Linear(self.internal_dim, output_dim)
        self.layer_norm = nn.LayerNorm(self.internal_dim)
        self.coeffs = nn.Parameter(torch.randn(self.internal_dim, complexity) * 0.05)
        self.exponents = nn.Parameter(torch.rand(self.internal_dim, complexity) * 1.5 + 0.25)
        self.base_funcs = [ torch.sin, torch.cos, torch.tanh, nn.SiLU(), nn.ReLU6() ]
        self.num_base_funcs = len(self.base_funcs)
        nn.init.xavier_uniform_(self.input_layer.weight)
        nn.init.zeros_(self.input_layer.bias)
        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
             try: x = torch.tensor(x, dtype=torch.float32, device=self.input_layer.weight.device)
             except Exception as e: raise TypeError(f"Input conversion failed: {e}")
        if x.dim() == 1: x = x.unsqueeze(0)
        if x.dim() == 0: x = x.unsqueeze(0).unsqueeze(0).expand(1, self.input_dim)
        if x.shape[1] != self.input_dim:
            if x.numel() == x.shape[0] * self.input_dim: x = x.view(x.shape[0], self.input_dim)
            else: raise ValueError(f"{self.__class__.__name__} expects input_dim={self.input_dim}, got shape {x.shape}")

        internal_x = self.input_layer(x)
        internal_x = self.layer_norm(internal_x)
        internal_x = torch.relu(internal_x)

        dynamic_sum = torch.zeros_like(internal_x)
        for i in range(self.complexity):
            func = self.base_funcs[i % self.num_base_funcs]
            coeff_i = self.coeffs[:, i].unsqueeze(0)
            exp_i = self.exponents[:, i].unsqueeze(0)
            term_input = internal_x * exp_i
            term_input_clamped = torch.clamp(term_input, -10.0, 10.0)
            try:
                term = coeff_i * func(term_input_clamped)
                term = torch.nan_to_num(term, nan=0.0, posinf=1e4, neginf=-1e4)
                dynamic_sum = dynamic_sum + term
            except RuntimeError as e: continue

        output = self.output_layer(dynamic_sum)
        output = torch.clamp(output, -100.0, 100.0)
        return output

class TauRLayer(nn.Module):
    """
    طبقة تحسب قيمة Tau مع تحسينات للاستقرار.
    """
    def __init__(self, input_dim, output_dim, epsilon=1e-6, alpha=0.1, beta=0.1):
        super().__init__()
        if not (isinstance(input_dim, int) and input_dim > 0 and isinstance(output_dim, int) and output_dim > 0):
             raise ValueError("input_dim and output_dim must be positive integers.")
        self.progress_transform = nn.Linear(input_dim, output_dim)
        self.risk_transform = nn.Linear(input_dim, output_dim)
        nn.init.xavier_uniform_(self.progress_transform.weight, gain=0.1)
        nn.init.zeros_(self.progress_transform.bias)
        nn.init.xavier_uniform_(self.risk_transform.weight, gain=0.1)
        nn.init.zeros_(self.risk_transform.bias)
        self.epsilon = epsilon; self.alpha = alpha; self.beta = beta
        self.min_denominator = 1e-5

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32, device=self.progress_transform.weight.device)
        if x.dim() == 1: x = x.unsqueeze(0)
        if x.dim() == 0: x = x.unsqueeze(0).unsqueeze(0).expand(1, self.progress_transform.in_features)
        if x.shape[1] != self.progress_transform.in_features:
             raise ValueError(f"{self.__class__.__name__} expects input_dim={self.progress_transform.in_features}, got shape {x.shape}")

        progress = torch.tanh(self.progress_transform(x))
        risk = torch.relu(self.risk_transform(x))
        numerator = progress + self.alpha
        denominator = risk + self.beta + self.epsilon
        denominator = torch.clamp(denominator, min=self.min_denominator)
        tau_output = numerator / denominator
        # --- إضافة Tanh وقص أضيق ---
        tau_output = torch.tanh(tau_output) # لضمان المخرج بين -1 و 1
        tau_output = torch.clamp(tau_output, min=-10.0, max=10.0) # قص إضافي بسيط

        return tau_output

class ChaosOptimizer(optim.Optimizer):
    def __init__(self, params, lr=0.001, sigma=10.0, rho=28.0, beta=8/3):
        if lr < 0.0: raise ValueError(f"Invalid learning rate: {lr}")
        defaults = dict(lr=lr, sigma=sigma, rho=rho, beta=beta)
        super().__init__(params, defaults)
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad(): loss = closure()
        for group in self.param_groups:
            lr, sigma, rho, beta_chaos = group['lr'], group['sigma'], group['rho'], group['beta']
            if not group['params']: continue
            for p in group['params']:
                if p.grad is None: continue
                grad = p.grad; param_state = p.data
                try:
                    dx = sigma * (grad - param_state)
                    dy = param_state * (rho - grad) - param_state
                    dz = param_state * grad - beta_chaos * param_state
                    chaotic_update = dx + dy + dz
                    if not torch.isfinite(chaotic_update).all(): continue
                    p.data.add_(chaotic_update, alpha=lr)
                except RuntimeError as e: continue
        return loss

class EvolvingNetwork(nn.Module):
    """
    شبكة عصبية يمكنها تطوير هيكلها بإضافة طبقات ديناميكيًا.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, initial_layers=1, max_layers=5):
        super().__init__()
        if not (isinstance(initial_layers, int) and initial_layers >= 1): raise ValueError("initial_layers must be at least 1.")
        self.input_dim = input_dim; self.hidden_dim = hidden_dim; self.output_dim = output_dim
        self.max_layers = max_layers; self.layers = nn.ModuleList()
        current_dim = input_dim
        for i in range(initial_layers):
            if not isinstance(current_dim, int) or current_dim <= 0: raise ValueError(f"Invalid input dim {current_dim} for layer {i}")
            if not isinstance(hidden_dim, int) or hidden_dim <= 0: raise ValueError(f"Invalid hidden dim {hidden_dim}")
            layer_block = nn.Sequential(
                nn.Linear(current_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
                TauRLayer(hidden_dim, hidden_dim) # <-- يستخدم النسخة المحسنة
            )
            self.layers.append(layer_block); current_dim = hidden_dim
        self.output_layer = nn.Linear(current_dim, output_dim)
        self.performance_history = deque(maxlen=30)
        self.evolution_threshold = 0.001

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
             x = torch.tensor(x, dtype=torch.float32, device=next(self.parameters()).device)
        if x.dim() == 1: x = x.unsqueeze(0)
        if x.dim() == 0: x = x.unsqueeze(0).unsqueeze(0).expand(1, self.input_dim)
        if x.shape[1] != self.input_dim:
             if x.numel() == x.shape[0] * self.input_dim: x = x.view(x.shape[0], self.input_dim)
             else: raise ValueError(f"{self.__class__.__name__} expects input_dim={self.input_dim}, got {x.shape}")

        current_x = x
        if not self.layers:
             print("Warning: EvolvingNetwork has no layers. Applying output layer directly.")
             return self.output_layer(current_x) # قد تحتاج لتعديل البعد هنا
        for layer_block in self.layers:
            current_x = layer_block(current_x)
            current_x = torch.relu(current_x) # ReLU بعد كل بلوك
        output = self.output_layer(current_x)
        return output

    def evolve(self):
        evolved = False
        if len(self.layers) >= self.max_layers: return evolved
        if len(self.performance_history) < self.performance_history.maxlen: return evolved
        recent_losses = list(self.performance_history)
        if len(recent_losses) < 5: return evolved
        improvement = np.mean(np.diff(recent_losses[-5:]))
        if improvement > -self.evolution_threshold:
            print(f"--- Evolving Network: Adding layer {len(self.layers) + 1}/{self.max_layers} (improvement: {improvement:.6f}) ---")
            try: device = next(self.parameters()).device
            except StopIteration: device = torch.device("cpu")
            try: last_layer_output_dim = self.layers[-1][-1].progress_transform.in_features
            except (IndexError, AttributeError): last_layer_output_dim = self.hidden_dim
            new_layer_input_dim = last_layer_output_dim
            new_layer_hidden_dim = self.hidden_dim
            if not isinstance(new_layer_input_dim, int) or new_layer_input_dim <= 0:
                 print(f"Error: Invalid input dimension ({new_layer_input_dim}) for new evolving layer. Evolution aborted.")
                 return False
            new_layer_block = nn.Sequential(
                nn.Linear(new_layer_input_dim, new_layer_hidden_dim), nn.LayerNorm(new_layer_hidden_dim), nn.ReLU(),
                TauRLayer(new_layer_hidden_dim, new_layer_hidden_dim)
            ).to(device)
            self.layers.append(new_layer_block)
            try:
                self.output_layer = nn.Linear(new_layer_hidden_dim, self.output_dim).to(device)
                print("Output layer rebuilt.")
            except Exception as e:
                 print(f"Error rebuilding output layer: {e}")
                 self.layers.pop(); return False
            self.performance_history.clear(); evolved = True
        return evolved

# --- نظام التعلم المعزز ---

class MathRLSystem:
    """
    نظام تعلم معزز متكامل (نسخة محسنة للاستقرار).
    """
    def __init__(self, state_dim, action_dim, hidden_dim=64, learning_rate=1e-4, # <-- LR أقل
                 gamma=0.99, buffer_size=10000, batch_size=64, update_target_every=10,
                 epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.996):
        self.state_dim = state_dim; self.action_dim = action_dim; self.gamma = gamma
        self.batch_size = batch_size; self.update_target_every = update_target_every
        self.memory = deque(maxlen=buffer_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using MathRLSystem on device: {self.device}")

        # استخدام بنية أبسط أولاً
        self.net = EvolvingNetwork(state_dim, hidden_dim, action_dim, initial_layers=1).to(self.device)
        self.target_net = deepcopy(self.net).to(self.device)
        self.target_net.eval()

        # --- استخدام AdamW ---
        print(f"Using AdamW optimizer with LR={learning_rate}")
        self.optimizer = optim.AdamW(self.net.parameters(), lr=learning_rate, amsgrad=True) # استخدام Amsgrad قد يساعد

        self.loss_fn = nn.HuberLoss(delta=1.0) # استخدام HuberLoss للاستقرار
        self.update_target_counter = 0
        self.epsilon = epsilon_start; self.epsilon_end = epsilon_end; self.epsilon_decay = epsilon_decay

    def adaptive_exploration(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def select_action(self, state, epsilon=None):
        current_epsilon = epsilon if epsilon is not None else self.epsilon
        if random.random() < current_epsilon:
            return random.randrange(self.action_dim)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            self.net.eval()
            with torch.no_grad(): q_values = self.net(state_tensor)
            self.net.train()
            return torch.argmax(q_values).item()

    def store_transition(self, state, action, reward, next_state, done):
        state = np.array(state, dtype=np.float32)
        next_state = np.array(next_state, dtype=np.float32)
        self.memory.append((state, action, float(reward), next_state, bool(done)))

    def _update_target_network_weights(self):
        try: self.target_net.load_state_dict(self.net.state_dict())
        except RuntimeError as e:
            print(f"Error updating target net weights (structure mismatch?): {e}. Recreating target net.")
            try: self.target_net = deepcopy(self.net); self.target_net.eval(); print("Target network rebuilt.")
            except Exception as deepcopy_e: print(f"FATAL: Failed to rebuild target network: {deepcopy_e}")

    def _rebuild_optimizer(self):
        print("Rebuilding optimizer...")
        try: last_lr = self.optimizer.param_groups[0]['lr']
        except Exception: last_lr = 1e-4 # قيمة افتراضية
        try:
            current_params = list(self.net.parameters())
            if not any(p.requires_grad for p in current_params): return False
            # --- استخدام AdamW ---
            self.optimizer = optim.AdamW(current_params, lr=last_lr, amsgrad=True)
            print(f"Optimizer rebuilt (AdamW) successfully with LR: {last_lr}")
            return True
        except Exception as e: print(f"Error rebuilding optimizer: {e}"); return False

    def update(self):
        if len(self.memory) < self.batch_size: return False
        if self.optimizer is None: return False

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones_bool = zip(*batch)

        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(np.array(dones_bool).astype(np.float32)).unsqueeze(1).to(self.device)

        # --- Q-Learning Update ---
        self.net.train()
        # --- قص المدخلات للشبكة ---
        states = torch.clamp(states, -10.0, 10.0)
        current_q_values = self.net(states)
        current_q = torch.gather(current_q_values, 1, actions)

        self.target_net.eval()
        with torch.no_grad():
            # --- قص المدخلات للشبكة الهدف ---
            next_states_clamped = torch.clamp(next_states, -10.0, 10.0)
            next_q_values_target = self.target_net(next_states_clamped)
            # --- قص قيم Q المستهدفة ---
            next_q_values_target = torch.clamp(next_q_values_target, -100.0, 100.0)
            next_q_target = next_q_values_target.max(1)[0].unsqueeze(1)

        expected_q = rewards + (1.0 - dones) * self.gamma * next_q_target
        # --- قص القيمة المتوقعة ---
        expected_q = torch.clamp(expected_q, -100.0, 100.0)

        loss = self.loss_fn(current_q, expected_q)
        if not torch.isfinite(loss):
            print(f"Warning: Non-finite loss detected ({loss.item()}). Skipping update.")
            # يمكنك إضافة تحليل هنا لمعرفة أي جزء يسبب المشكلة
            # print("Current Q:", current_q.min().item(), current_q.max().item(), current_q.mean().item())
            # print("Expected Q:", expected_q.min().item(), expected_q.max().item(), expected_q.mean().item())
            return False

        self.optimizer.zero_grad()
        loss.backward()
        # --- قص التدرجات بقوة أكبر ---
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=0.5)
        self.optimizer.step()

        # --- تطور الشبكة وتحديث الشبكة الهدف ---
        self.net.performance_history.append(loss.item())
        structure_changed = self.net.evolve()

        if structure_changed:
            if self._rebuild_optimizer():
                 print("Recreating target network after structure evolution.")
                 try: self.target_net = deepcopy(self.net); self.target_net.eval(); self.update_target_counter = 0
                 except Exception as e: print(f"Error deepcopying evolved network: {e}")
            else: print("ERROR: Failed to rebuild optimizer after evolution.")
        else:
            self.update_target_counter += 1
            if self.update_target_counter % self.update_target_every == 0:
                self._update_target_network_weights()
        return True


    def train(self, env, episodes=1000, max_steps_per_episode=200, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.996):
        self.epsilon = epsilon_start; self.epsilon_end = epsilon_end; self.epsilon_decay = epsilon_decay
        rewards_history = []

        for ep in range(episodes):
            state = env.reset()
            if not isinstance(state, np.ndarray): state = np.array(state, dtype=np.float32)
            total_reward = 0; episode_steps = 0; done = False

            while not done and episode_steps < max_steps_per_episode:
                action = self.select_action(state)
                next_state, reward, done, _ = env.step(action)
                if not isinstance(next_state, np.ndarray): next_state = np.array(next_state, dtype=np.float32)
                self.store_transition(state, action, reward, next_state, done)
                updated = False
                if len(self.memory) >= self.batch_size:
                    # تحديث عدة مرات لكل خطوة لزيادة استخدام البيانات
                    for _ in range(2): # مثال: تحديث مرتين
                         if len(self.memory) >= self.batch_size:
                              updated = self.update()
                         else: break
                state = next_state; total_reward += reward; episode_steps += 1

            self.adaptive_exploration()
            rewards_history.append(total_reward)
            if (ep + 1) % 10 == 0:
                print(f"Episode {ep+1}/{episodes}, Reward: {total_reward:.2f}, Steps: {episode_steps}, Epsilon: {self.epsilon:.3f}")

        # --- رسم الأداء بعد التدريب ---
        plt.figure(figsize=(10, 6))
        plt.plot(rewards_history, label='Episode Reward')
        if len(rewards_history) >= 20:
             moving_avg = np.convolve(rewards_history, np.ones(20)/20, mode='valid')
             plt.plot(np.arange(len(moving_avg)) + 19, moving_avg, label='20-Ep MA', color='red', alpha=0.7)
        plt.title(f'{self.__class__.__name__} Training Performance (Stability Enhanced)')
        plt.xlabel('Episode'); plt.ylabel('Total Reward per Episode')
        plt.legend(); plt.grid(True, linestyle=':'); plt.tight_layout()
        plt.savefig("mathrl_stable_v2_training.png"); plt.show()

# --- بيئة اختبار بسيطة ---
class SimpleCorridorEnv:
    def __init__(self, size=10):
        if size < 2: raise ValueError("Size must be at least 2")
        self.size = size; self.agent_pos = 0; self.goal_pos = size - 1
        self.action_space = type('ActionSpace', (), {'n': 2, 'sample': lambda: random.randint(0, 1)})()
        self.observation_space = type('ObservationSpace', (), {'shape': (1,), 'dtype': np.float32})()
        print(f"SimpleCorridorEnv initialized with size {size}")
    def reset(self):
        self.agent_pos = 0
        return np.array([self.agent_pos], dtype=self.observation_space.dtype)
    def step(self, action):
        if action == 0: self.agent_pos = max(0, self.agent_pos - 1)
        elif action == 1: self.agent_pos = min(self.size - 1, self.agent_pos + 1)
        else: raise ValueError(f"Invalid action: {action}.")
        if self.agent_pos == self.goal_pos: reward = 1.0; done = True
        else: reward = -0.05; done = False
        next_state = np.array([self.agent_pos], dtype=self.observation_space.dtype)
        return next_state, reward, done, {}
    def render(self, mode='human'):
        if mode == 'human':
            corridor = ['_'] * self.size; corridor[self.agent_pos] = 'A'
            if self.agent_pos != self.goal_pos: corridor[self.goal_pos] = 'G'
            print(f"|{' '.join(corridor)}| Pos: {self.agent_pos}")

# --- التشغيل والاختبار ---
if __name__ == "__main__":
    random.seed(43); np.random.seed(43); torch.manual_seed(43) # تغيير البذرة
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(43)

    env = SimpleCorridorEnv(size=15)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # --- استخدام AdamW ومعدل تعلم أقل وبنية أولية أبسط ---
    agent = MathRLSystem(state_dim=state_dim, action_dim=action_dim,
                         hidden_dim=32,         # <-- أصغر
                         learning_rate=5e-5,   # <-- أقل بكثير
                         buffer_size=10000,     # <-- أكبر قليلاً
                         batch_size=64,         # <-- أكبر
                         update_target_every=20,# <-- تحديث أقل للهدف
                         epsilon_decay=0.995)

    print("Starting MathRLSystem training (Stability Enhanced)...")
    agent.train(env, episodes=300, max_steps_per_episode=150) # زيادة الحلقات قليلاً
    print("Training finished.")

    print("\nTesting the trained agent...")
    state = env.reset()
    done = False; total_reward_test = 0; steps_test = 0; max_test_steps = 100
    env.render()
    while not done and steps_test < max_test_steps:
        action = agent.select_action(state, epsilon=0.01)
        state, reward, done, _ = env.step(action)
        env.render()
        total_reward_test += reward; steps_test += 1
        if done: print("Goal Reached!"); break
    if not done and steps_test == max_test_steps: print("Max steps reached during testing.")
    print(f"Test finished. Total reward: {total_reward_test:.2f}, Steps: {steps_test}")

'''
وصف الفكرة :
الكود المقدم يُنفذ نظامًا للتعلم المعزز (Reinforcement Learning) مع ميزات متقدمة لتحسين الاستقرار والأداء، ويُسمى MathRLSystem . إليك تفاصيل عمله وفكرته الأساسية:

الهدف الرئيسي:
بناء وكيل ذكاء اصطناعي قادر على التعلم في بيئة ديناميكية (مثل متاهة بسيطة) باستخدام خوارزمية Q-Learning المُحسَّنة، مع دمج تقنيات لتحسين الاستقرار مثل:

الشبكات العصبية الديناميكية (التي تتطور ذاتيًا بإضافة طبقات جديدة عند الحاجة).
تحسينات رياضية (مثل استخدام دوال تنشيط غير خطية وتحسين AdamW).
ميكانيكيات استقرار (مثل تقييد التدرجات وقص القيم).
المكونات الرئيسية:
DynamicMathUnit :
وحدة شبكة عصبية تُنفذ تحويلات رياضية معقدة باستخدام دوال مثل sin, cos, tanh.
تستخدم معاملات قابلة للتعلم (coeffs, exponents) لتعديل الإشارات الديناميكية.
TauRLayer :
طبقة مخصصة تحسب قيمة Tau التي تمثل مزيجًا من "التقدم" و"المخاطرة" في عملية التعلم.
تستخدم tanh للحفاظ على القيم ضمن نطاق محدود.
ChaosOptimizer :
مُحسِّن مخصص يعتمد على نظام لورينتز الفوضوي (Lorenz System) لتحديث الأوزان، مما يساعد في الهروب من الحدود الدنيا المحلية.
EvolvingNetwork :
شبكة عصبية تتطور ذاتيًا بإضافة طبقات جديدة إذا لاحظت أن الأداء يتوقف عن التحسن.
تراقب سجل الخسائر (performance_history) لاتخاذ قرار التطور.
MathRLSystem :
النظام الرئيسي للتعلم المعزز، يعتمد على:
Q-Learning مع شبكة عصبية لتقدير قيم Q.
الشبكة الهدف (Target Network) لتحديث مستقر.
ذاكرة التجارب (Replay Buffer) لتخزين الخبرات.
استكشاف متكيف (Epsilon-Greedy) مع تناقص ε تدريجي.
SimpleCorridorEnv :
بيئة اختبار بسيطة حيث يتحرك الوكيل في ممر لوصول إلى الهدف، مع مكافآت سلبية صغيرة لكل خطوة ومكافأة كبيرة عند الوصول.
خطوات العمل:
التهيئة :
تهيئة الشبكة العصبية (EvolvingNetwork) مع طبقة أولية.
ضبط المعلمات مثل معدل التعلم (learning_rate=5e-5) وحجم الذاكرة (buffer_size=10000).
التدريب :
في كل حلقة تدريب:
يختار الوكيل إجراءً (يسار/يمين) بناءً على ε-Greedy.
يخزن التجربة (الحالة، الإجراء، المكافأة، الحالة التالية) في الذاكرة.
يحدّث الشبكة باستخدام عينات عشوائية من الذاكرة (Experience Replay).
يتطور الشكل الهندسي للشبكة إذا لزم الأمر (إضافة طبقات).
التحديثات :
تحديث الشبكة الهدف كل update_target_every خطوة.
استخدام AdamW مع amsgrad=True لتحسين الاستقرار.
تقييد التدرجات (clip_grad_norm_) لمنع الانفجار.
الاختبار :
بعد التدريب، يُختبر الأداء في البيئة مع ε منخفض (0.01 لتجنب الاستكشاف العشوائي).
التحسينات الرئيسية للاستقرار:
استخدام Huber Loss بدلًا من MSE لتقليل تأثير القيم الشاذة.
تقييد القيم المُخرجة من الشبكات (clamp) لمنع الانفجار.
تفعيل gradient clipping أثناء التحديث.
استخدام AdamW مع تصحيح التدرجات (Amsgrad).
البيئة (SimpleCorridorEnv):
الهدف : الوصول إلى نهاية الممر (الموقع size-1).
الإجراءات : 0 (التحرك يسارًا)، 1 (التحرك يمينًا).
المكافآت :
-0.05 لكل خطوة (تشجيع على السرعة).
+1 عند الوصول إلى الهدف.
الناتج المتوقع:
رسم بياني لأداء التدريب مع متوسط متحرك (Moving Average).
طباعة تفاصيل الأداء في كل حلقة (المكافأة، عدد الخطوات، قيمة ε).
عرض مرئي لحركة الوكيل في مرحلة الاختبار.
الخلاصة:
الكود يُظهر نظامًا متقدمًا للتعلم المعزز مع دمج مفاهيم مثل التطور الذاتي للشبكات العصبية، والاستخدام الفوضوي للمُحسِّنات، وآليات الاستقرار الرياضي، مما يجعله مناسبًا لحل مسائل التحكم الديناميكي المعقدة.
'''