# -*- coding: utf-8 -*-
# Evolving Equation Network (EvoEqNet) - 

"""
EvoEqNet: نظام تعلم  مبتكر يعتمد على معادلات رياضية تتطوّر مع التدريب
تم تطويره بواسطة: [باسل يحيى عبدالله/ العراق/ الموصل ]
تاريخ الإصدار: [20/4/2025]
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from copy import deepcopy
import random
import os
import json

# --- المكونات الأساسية للشبكة ---

class DynamicEquationUnit(nn.Module):
    """
    وحدة رياضية تمثل معادلة ديناميكية قابلة للتعلم.
    **مع تحسينات للاستقرار العددي.**
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

class EvolvingEquationLayer(nn.Module):
    """
    طبقة تحتوي على وحدات معادلات ديناميكية ويمكنها النمو.
    """
    def __init__(self, input_dim, output_dim, initial_units=1, growth_threshold=0.01, max_units=20):
        super().__init__()
        if not (isinstance(initial_units, int) and initial_units >= 1): raise ValueError("initial_units must be at least 1.")
        self.input_dim = input_dim; self.output_dim = output_dim; self.initial_units = initial_units; self.max_units = max_units
        self.units = nn.ModuleList([DynamicEquationUnit(input_dim, output_dim) for _ in range(initial_units)])
        self.growth_threshold = growth_threshold; self.performance_history = deque(maxlen=60)
    def get_num_units(self): return len(self.units)
    def add_unit(self):
        if len(self.units) >= self.max_units: return False
        print(f"+++ Evolving Layer ({self.input_dim}->{self.output_dim}): Adding unit {len(self.units) + 1}/{self.max_units} +++")
        try: device = next(self.parameters()).device
        except StopIteration: device = torch.device("cpu")
        new_unit = DynamicEquationUnit(self.input_dim, self.output_dim).to(device); self.units.append(new_unit)
        return True
    def check_performance_and_grow(self, current_loss):
        if not np.isfinite(current_loss): return False
        self.performance_history.append(current_loss)
        if len(self.performance_history) < self.performance_history.maxlen: return False
        evolved = False; recent_losses = list(self.performance_history)
        if len(recent_losses) > 1: improvement = np.mean(np.diff(recent_losses))
        else: improvement = 0
        if improvement > -self.growth_threshold:
            if self.add_unit(): self.performance_history.clear(); evolved = True
        return evolved
    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            try: device = next(self.parameters()).device
            except StopIteration: device = torch.device("cpu")
            x = torch.tensor(x, dtype=torch.float32, device=device)
        if x.dim() == 1: x = x.unsqueeze(0)
        unit_outputs = [unit(x) for unit in self.units]
        valid_outputs = [out for out in unit_outputs if isinstance(out, torch.Tensor) and torch.isfinite(out).all()]
        if not valid_outputs: return torch.zeros((x.shape[0], self.output_dim), device=x.device, dtype=torch.float32)
        try:
            combined_output = torch.stack(valid_outputs).mean(dim=0)
            combined_output = torch.clamp(combined_output, -100.0, 100.0)
        except RuntimeError as e: combined_output = valid_outputs[0]
        return combined_output

class EvoEqNet(nn.Module):
    """
    الشبكة الرئيسية مع تصحيح جذري لـ add_layer.
    """
    def __init__(self, input_dim, hidden_dims, output_dim, dropout_rate=0.2, max_layers=10):
        super().__init__()
        self.input_dim = input_dim
        self.initial_hidden_dims = list(hidden_dims) # حفظ الأبعاد الأولية
        self.output_dim = output_dim
        self.max_layers = max_layers
        self.dropout_rate = dropout_rate

        self.hidden_layers = nn.ModuleList() # قائمة للطبقات المخفية فقط
        current_dim = input_dim
        for i, h_dim in enumerate(self.initial_hidden_dims):
            if not (isinstance(current_dim, int) and current_dim > 0 and isinstance(h_dim, int) and h_dim > 0):
                 raise ValueError(f"Invalid dimensions for hidden layer {i}: input={current_dim}, output={h_dim}")
            layer = EvolvingEquationLayer(current_dim, h_dim, initial_units=1)
            self.hidden_layers.append(layer)
            current_dim = h_dim

        self.output_layer = nn.Linear(current_dim, output_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.network_performance_history = deque(maxlen=100)
        self.layer_evolution_threshold = 0.002

    def get_architecture_info(self):
        current_hidden_dims = [layer.output_dim for layer in self.hidden_layers]
        info = {'input_dim': self.input_dim, 'hidden_dims': current_hidden_dims, 'output_dim': self.output_dim,
                'layer_units': [layer.get_num_units() for layer in self.hidden_layers],
                'max_layers': self.max_layers, 'dropout_rate': self.dropout.p}
        return info

    @classmethod
    def from_architecture_info(cls, info):
        # إنشاء الشبكة بالأبعاد المحفوظة في hidden_dims
        net = cls(info['input_dim'], info['hidden_dims'], info['output_dim'],
                  info.get('dropout_rate', 0.2), info.get('max_layers', 10))
        # تعديل عدد الوحدات في الطبقات المخفية
        if len(net.hidden_layers) == len(info['layer_units']):
             device = next(net.parameters()).device if list(net.parameters()) else torch.device("cpu")
             for i, num_units_saved in enumerate(info['layer_units']):
                 layer = net.hidden_layers[i]; current_units = layer.get_num_units()
                 units_to_add = num_units_saved - current_units
                 if units_to_add > 0:
                      print(f"Loading architecture: Adding {units_to_add} units to hidden layer {i}")
                      for _ in range(units_to_add):
                           new_unit = DynamicEquationUnit(layer.input_dim, layer.output_dim).to(device)
                           layer.units.append(new_unit)
                 elif units_to_add < 0: print(f"Warning: Saved model has fewer units in hidden layer {i}.")
        else: print("Warning: Hidden layer count mismatch during loading.")
        return net

    # ====> الدالة add_layer المعدلة (إصدار أبسط) <====
    def add_layer(self):
        """
        إضافة طبقة تطورية جديدة في نهاية الطبقات المخفية (قبل طبقة الإخراج).
        """
        if len(self.hidden_layers) >= self.max_layers: return False

        print(f"*** Evolving Network Architecture: Adding hidden layer {len(self.hidden_layers) + 1}/{self.max_layers} ***")
        try: device = next(self.parameters()).device
        except StopIteration: device = torch.device("cpu")

        # مدخل الطبقة الجديدة هو مخرج الطبقة المخفية الأخيرة الحالية
        if self.hidden_layers: new_layer_input_dim = self.hidden_layers[-1].output_dim
        else: new_layer_input_dim = self.input_dim # إذا لم تكن هناك طبقات مخفية

        # مخرج الطبقة الجديدة هو نفس مدخلها (أو أول بعد مخفي للحفاظ على التسلسل)
        new_layer_output_dim = new_layer_input_dim

        if not isinstance(new_layer_input_dim, int) or new_layer_input_dim <= 0: print(f"Error: Invalid input dim ({new_layer_input_dim})."); return False
        if not isinstance(new_layer_output_dim, int) or new_layer_output_dim <= 0: print(f"Error: Invalid output dim ({new_layer_output_dim})."); return False

        print(f"Adding new hidden layer at index {len(self.hidden_layers)} with dims: {new_layer_input_dim} -> {new_layer_output_dim}")
        new_layer = EvolvingEquationLayer(new_layer_input_dim, new_layer_output_dim).to(device)
        self.hidden_layers.append(new_layer) # الإضافة للنهاية

        # إعادة بناء طبقة الإخراج النهائية لتقبل المخرج الجديد
        print(f"Rebuilding output layer to accept input dim: {new_layer_output_dim}")
        self.output_layer = nn.Linear(new_layer_output_dim, self.output_dim).to(device)
        nn.init.xavier_uniform_(self.output_layer.weight); nn.init.zeros_(self.output_layer.bias)

        return True
    # ====> نهاية add_layer المعدلة <====

    def evolve_structure(self, validation_loss):
        evolved = False
        if not np.isfinite(validation_loss): return evolved
        self.network_performance_history.append(validation_loss)
        if len(self.network_performance_history) < self.network_performance_history.maxlen: return evolved
        recent_losses = list(self.network_performance_history)
        if len(recent_losses) > 50:
            improvement = np.mean(np.diff(recent_losses[-50:]))
            if improvement > -self.layer_evolution_threshold:
                if self.add_layer(): evolved = True; self.network_performance_history.clear()
        return evolved

    def forward(self, x):
        current_x = x
        for i, layer in enumerate(self.hidden_layers): # المرور عبر الطبقات المخفية فقط
            current_x = layer(current_x)
            current_x = torch.tanh(current_x)
            current_x = self.dropout(current_x)
        output = self.output_layer(current_x) # تطبيق طبقة الإخراج
        return output

# --- محسن AdamW (أكثر استقرارًا) ---
# class ChaosOptimizer(optim.Optimizer): ...

# --- نظام التدريب المتكامل ---
class TrainingSystem:
    """ نظام تدريب يدير تدريب وتقييم وتطور شبكة EvoEqNet. """
    def __init__(self, input_dim, hidden_dims, output_dim, use_chaos_optimizer=False):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); print(f"Using device: {self.device}")
        self.input_dim = input_dim; self.initial_hidden_dims = list(hidden_dims); self.output_dim = output_dim
        self.use_chaos_optimizer = use_chaos_optimizer
        self.net = EvoEqNet(input_dim, list(hidden_dims), output_dim).to(self.device)
        self.optimizer = None; self.loss_fn = nn.HuberLoss(delta=1.0); self.scheduler = None
        self.best_loss = float('inf')
        self.model_save_path = 'best_evoeqnet_model.pth'; self.arch_save_path = 'best_evoeqnet_arch.json'
        self.data_mean_X = None; self.data_std_X = None; self.data_mean_y = None; self.data_std_y = None
        self._rebuild_optimizer() # تهيئة المحسن والمجدول

    def _normalize_X(self, X_tensor):
        if not isinstance(X_tensor, torch.Tensor): X_tensor = torch.tensor(X_tensor, dtype=torch.float32, device=self.device)
        if self.data_mean_X is None:
             self.data_mean_X = torch.mean(X_tensor, dim=0, keepdim=True)
             self.data_std_X = torch.std(X_tensor, dim=0, keepdim=True); self.data_std_X[self.data_std_X < 1e-8] = 1.0
        return (X_tensor - self.data_mean_X) / self.data_std_X
    def _normalize_y(self, y_tensor):
        if not isinstance(y_tensor, torch.Tensor): y_tensor = torch.tensor(y_tensor, dtype=torch.float32, device=self.device)
        if self.data_mean_y is None:
             self.data_mean_y = torch.mean(y_tensor, dim=0, keepdim=True)
             self.data_std_y = torch.std(y_tensor, dim=0, keepdim=True); self.data_std_y[self.data_std_y < 1e-8] = 1.0
        return (y_tensor - self.data_mean_y) / self.data_std_y
    def _denormalize_y(self, y_norm_tensor):
        if self.data_mean_y is None or self.data_std_y is None: return y_norm_tensor
        std_dev = self.data_std_y.to(y_norm_tensor.device); mean_val = self.data_mean_y.to(y_norm_tensor.device)
        return y_norm_tensor * std_dev + mean_val

    def _rebuild_optimizer(self):
        print("Rebuilding optimizer...")
        current_lr = 0.001 # إعادة تعيين لمعدل التعلم الافتراضي
        if self.optimizer and self.optimizer.param_groups: current_lr = self.optimizer.param_groups[0]['lr']
        elif self.scheduler and hasattr(self.scheduler, 'get_last_lr') and self.scheduler.get_last_lr(): current_lr = self.scheduler.get_last_lr()[0]
        try:
            current_params = list(self.net.parameters())
            if not any(p.requires_grad for p in current_params): print("Warning: No trainable parameters."); return False
            if self.use_chaos_optimizer: self.optimizer = ChaosOptimizer(current_params, lr=current_lr)
            else: self.optimizer = optim.AdamW(current_params, lr=current_lr, weight_decay=1e-4)
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=50, verbose=False)
            print(f"Optimizer rebuilt ({'Chaos' if self.use_chaos_optimizer else 'AdamW'}) with LR: {current_lr:.6f}")
            return True
        except Exception as e: print(f"Unexpected error rebuilding optimizer: {e}"); return False

    def train(self, X, y, epochs=3000, validation_split=0.2):
        X_tensor = torch.FloatTensor(X).to(self.device); y_tensor = torch.FloatTensor(y).to(self.device)
        if X_tensor.shape[0] < 5: print("Error: Dataset too small."); return
        dataset_size = len(X_tensor); split = max(1, min(dataset_size - 1, int(np.floor(validation_split * dataset_size))))
        indices = list(range(dataset_size)); np.random.shuffle(indices); train_indices, val_indices = indices[split:], indices[:split]
        X_train_orig, X_val_orig = X_tensor[train_indices], X_tensor[val_indices]
        y_train_orig, y_val_orig = y_tensor[train_indices], y_tensor[val_indices]
        X_train = self._normalize_X(X_train_orig); y_train = self._normalize_y(y_train_orig)
        X_val = self._normalize_X(X_val_orig); y_val = self._normalize_y(y_val_orig)
        print(f"Training on {len(X_train)} samples, validating on {len(X_val)} samples.")
        train_loss_history = []; val_loss_history = []
        structure_changed_since_opt_rebuild = False

        for epoch in range(epochs):
            self.net.train()
            if structure_changed_since_opt_rebuild:
                if not self._rebuild_optimizer(): break
                structure_changed_since_opt_rebuild = False
            try:
                pred_train = self.net(X_train)
                if not torch.isfinite(pred_train).all(): print(f"Epoch {epoch}: Non-finite predictions (train). Skipping."); continue
                loss_train = self.loss_fn(pred_train, y_train)
                if not torch.isfinite(loss_train): print(f"Epoch {epoch}: Non-finite training loss ({loss_train.item()}). Skipping."); continue
                self.optimizer.zero_grad(); loss_train.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
                self.optimizer.step()
            except RuntimeError as e: print(f"RuntimeError during training step {epoch}: {e}"); continue
            train_loss_history.append(loss_train.item())

            self.net.eval()
            with torch.no_grad():
                pred_val = self.net(X_val)
                if not torch.isfinite(pred_val).all(): print(f"Epoch {epoch}: Non-finite predictions (val)."); val_loss_to_append = float('inf')
                else: loss_val = self.loss_fn(pred_val, y_val); val_loss_to_append = loss_val.item() if torch.isfinite(loss_val) else float('inf')
                if not np.isfinite(val_loss_to_append): print(f"Epoch {epoch}: Non-finite validation loss.")
                val_loss_history.append(val_loss_to_append)
            if np.isfinite(val_loss_to_append): self.scheduler.step(val_loss_to_append)

            layer_evolved = False; network_evolved = False
            if np.isfinite(loss_train.item()): layer_evolved = any(layer.check_performance_and_grow(loss_train.item()) for layer in self.net.hidden_layers if isinstance(layer, EvolvingEquationLayer))
            if np.isfinite(val_loss_to_append): network_evolved = self.net.evolve_structure(val_loss_to_append)
            if layer_evolved or network_evolved: structure_changed_since_opt_rebuild = True

            if np.isfinite(val_loss_to_append) and val_loss_to_append < self.best_loss:
                self.best_loss = val_loss_to_append
                try:
                     arch_info = self.net.get_architecture_info()
                     with open(self.arch_save_path, 'w') as f: json.dump(arch_info, f)
                     torch.save(self.net.state_dict(), self.model_save_path)
                except Exception as save_e: print(f"Error saving model/architecture: {save_e}")

            if epoch % 100 == 0 or epoch == epochs - 1:
                current_lr = self.optimizer.param_groups[0]['lr'] if self.optimizer and self.optimizer.param_groups else 'N/A'
                try: units_info = self.net.hidden_layers[0].get_num_units() if self.net.hidden_layers else "N/A"
                except IndexError: units_info = "N/A"
                print(f'Epoch {epoch:04d}/{epochs} | Train Loss: {loss_train.item():.6f} | Val Loss: {val_loss_to_append:.6f} | LR: {current_lr:.6f} | Layers: {len(self.net.hidden_layers)} | Units[0]: {units_info}')

        plt.figure(figsize=(12, 6))
        valid_train_loss = [l for l in train_loss_history if np.isfinite(l)]; valid_val_loss = [l for l in val_loss_history if np.isfinite(l)]
        plt.plot(valid_train_loss, label='Training Loss', alpha=0.8); plt.plot(valid_val_loss, label='Validation Loss', alpha=0.8)
        plt.title('EvoEqNet Training & Validation Loss Evolution (Corrected Dim Evolution)'); plt.xlabel('Epoch'); plt.ylabel('Loss (Huber)'); plt.yscale('log')
        plt.legend(); plt.grid(True, linestyle='--'); plt.tight_layout(); plt.savefig("evoeqnet_dim_corrected_v3_training_loss.png"); plt.show()

    def load_best_model(self):
        if os.path.exists(self.model_save_path) and os.path.exists(self.arch_save_path):
            try:
                with open(self.arch_save_path, 'r') as f: arch_info = json.load(f)
                print(f"Loading architecture: {arch_info}")
                self.net = EvoEqNet.from_architecture_info(arch_info).to(self.device)
                state_dict = torch.load(self.model_save_path, map_location=self.device)
                load_result = self.net.load_state_dict(state_dict, strict=False)
                print(f"Model state_dict loaded. Missing keys: {load_result.missing_keys}, Unexpected keys: {load_result.unexpected_keys}")
                self.net.eval()
                if not self._rebuild_optimizer(): print("Warning: Failed to rebuild optimizer after loading.")
                print(f"Successfully loaded best model and architecture.")
            except Exception as e:
                print(f"Error loading model or architecture: {e}. Re-initializing.")
                self.net = EvoEqNet(self.input_dim, self.initial_hidden_dims, self.output_dim).to(self.device)
                self._rebuild_optimizer()
        else: print(f"No saved model/architecture found.")


# --- التشغيل والاختبار ---
if __name__ == "__main__":
    seed = 42; random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

    X = np.linspace(-5, 5, 500).reshape(-1, 1)
    y = np.sin(X * 1.5) * np.exp(-X**2 / 8.0) + np.cos(X * 2.5) * 0.3 + np.random.normal(0, 0.05, X.shape)
    y = y.reshape(-1, 1)

    input_dim = X.shape[1]; output_dim = y.shape[1]
    system = TrainingSystem(input_dim=input_dim, hidden_dims=[16], output_dim=output_dim, use_chaos_optimizer=False)

    system.train(X, y, epochs=1500, validation_split=0.2)

    print("\nLoading best model for testing...")
    system.load_best_model()

    test_X = np.linspace(-7, 7, 1000).reshape(-1, 1)
    test_X_tensor = torch.FloatTensor(test_X).to(system.device)
    if system.data_mean_X is None: test_X_norm = test_X_tensor
    else: test_X_norm = system._normalize_X(test_X_tensor)

    with torch.no_grad():
        system.net.eval()
        pred_y_norm = system.net(test_X_norm)
        pred_y = system._denormalize_y(pred_y_norm).cpu().numpy()

    plt.figure(figsize=(14, 7))
    plt.scatter(X, y, c='blue', alpha=0.3, s=15, label='Training Data')
    plt.plot(test_X, pred_y, 'r-', lw=2.5, label=f'EvoEqNet Prediction (Layers: {len(system.net.hidden_layers)})')
    y_true_smooth = np.sin(test_X * 1.5) * np.exp(-test_X**2 / 8.0) + np.cos(test_X * 2.5) * 0.3
    plt.plot(test_X, y_true_smooth, 'g--', lw=1.5, label='True Underlying Function', alpha=0.7)
    plt.title('EvoEqNet (Corrected Dim Evolution v2): Complex Function Approximation')
    plt.xlabel('Input (X)'); plt.ylabel('Output (Y)')
    plt.xlim(test_X.min(), test_X.max()); plt.ylim(min(y.min(), pred_y.min()) - 0.5, max(y.max(), pred_y.max()) + 0.5)
    plt.legend(); plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout(); plt.savefig("evoeqnet_dim_corrected_v3_prediction.png"); plt.show()

    '''
    الكود المقدم يُظهر نظامًا متقدمًا للشبكات العصبية الديناميكية القابلة للتطور (EvoEqNet ) مع تركيز على الاستقرار العددي والقدرة على التكيف مع تعقيدات الدوال الرياضية. إليك تفاصيله:

الهدف الرئيسي:
بناء شبكة عصبية قادرة على:

التطور التلقائي بإضافة طبقات ووحدات جديدة عند استشعار توقف التحسن.
التمثيل الديناميكي للعلاقات الرياضية المعقدة باستخدام وحدات معادلات قابلة للتعلم.
التعامل مع عدم الاستقرار العددي عبر تقنيات مثل التقيييد (clamp) ودوال التنشيط المُحسَّنة.
المكونات الرئيسية:
DynamicEquationUnit :
وحدة رياضية تدمج دوالًا غير خطية (مثل sin, cos, SiLU) مع معاملات قابلة للتعلم.
تستخدم تحويلات مُعقدة (مثل الأسس والتركيب الدالي) لتمثيل العلاقات الديناميكية.
EvolvingEquationLayer :
طبقة تحتوي على عدة وحدات DynamicEquationUnit وتُضيف وحدات جديدة عند توقف التحسن.
تراقب سجل الخسائر (performance_history) لاتخاذ قرار التطور.
EvoEqNet :
الشبكة الرئيسية التي تضم:
طبقات مخفية قابلة للزيادة (hidden_layers).
آلية لإضافة طبقات جديدة (add_layer) عند استقرار الأداء.
Dropout لمنع overfitting.
TrainingSystem :
نظام تدريب متكامل يدعم:
التطبيع التلقائي للبيانات (_normalize_X, _normalize_y).
الحفظ التلقائي لأفضل نموذج.
الاستعادة للهيكل المعماري والبارامترات.
استخدام AdamW أو ChaosOptimizer (اختياري).
خطوات العمل:
التدريب :
تُطبَّق دورة تدريبية مع:
Huber Loss للاستقرار.
ReduceLROnPlateau لتعديل معدل التعلم.
التطور التلقائي للهيكل (إضافة طبقات/وحدات).
التطور الهيكلي :
تُضاف طبقات جديدة إذا لاحظ النظام توقف تحسن الخسارة (evolve_structure).
تُضاف وحدات جديدة داخل الطبقات (add_unit) عند استقرار الأداء.
التنبؤ :
بعد التدريب، يُستخدم النموذج الأفضل (المُخزَّن) للتنبؤ بدقة أعلى.
التحسينات الرئيسية للاستقرار:
تقيييد القيم (clamp) في كل مرحلة لمنع الانفجار.
التطبيع التلقائي للإدخالات والإخراجات.
استخدام AdamW مع weight_decay لتحسين التقارب.
التحكم في التدرجات (clip_grad_norm_).
مثال التشغيل:
البيانات : دالة مركبة (sin و cos مع ضوضاء).
النتيجة : يتعلم النموذج تقريب الدالة حتى خارج نطاق التدريب (test_X من -7 إلى 7).
الناتج المتوقع:
رسم بياني للخسارة أثناء التدريب (تدريب vs. تحقق).
مقارنة مرئية بين التنبؤات والدالة الحقيقية.
الخلاصة:
الكود يُظهر نظامًا ذكيًا للشبكات العصبية يتكيف ذاتيًا مع تعقيد المهام عبر التطور الهيكلي، مع التركيز على الاستقرار العددي وقابلية التطبيق على مسائل التقريب الرياضي
    '''