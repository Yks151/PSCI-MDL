import os
import re
import time  # 确保导入了time模块
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import gc
from diffusion import DiffusionModel  # 导入扩散模型

# 设置随机种子
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True

# ================== 辅助类定义 ==================
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
    
    def forward(self, logits, targets):
        log_probs = F.log_softmax(logits, dim=-1)
        nll_loss = -log_probs.gather(dim=-1, index=targets.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -log_probs.mean(dim=-1)
        loss = (1.0 - self.smoothing) * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

class EarlyStopping:
    def __init__(self, patience=10, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

# ================== 数据加载器 ==================
class NuclearDataLoader:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.required_timesteps = 61
        self.required_params = 88
        self.param_groups = {
            'flow': (0, 20),
            'temperature': (20, 40),
            'pressure': (40, 60),
            'level': (60, 80),
            'other': (80, 88)
        }
    
    def _extract_numeric_data(self, content):
        numeric_pattern = r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?'
        numbers = re.findall(numeric_pattern, content)
        numeric_values = []
        for num in numbers:
            try:
                clean_num = num.replace(',', '.')
                numeric_values.append(float(clean_num))
            except:
                continue
        return numeric_values
    
    def _load_and_validate_file(self, filepath):
        filename = os.path.basename(filepath)
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            numeric_values = self._extract_numeric_data(content)
            total_values_needed = self.required_timesteps * self.required_params
            
            if len(numeric_values) < total_values_needed:
                # 尝试通过插值填充缺失数据
                print(f"Warning: Only {len(numeric_values)} numbers found in {filename}, need {total_values_needed}. Attempting interpolation.")
                x_orig = np.linspace(0, 1, len(numeric_values))
                x_new = np.linspace(0, 1, total_values_needed)
                interp_func = interpolate.interp1d(
                    x_orig,
                    numeric_values,
                    kind='linear',
                    fill_value="extrapolate"
                )
                numeric_values = interp_func(x_new)
            
            data_matrix = np.array(numeric_values[:total_values_needed]).reshape(
                self.required_timesteps, self.required_params
            )
            fault_type = self._detect_fault_type(filename, data_matrix)
            return data_matrix, fault_type
            
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
            return None, None
    
    def _detect_fault_type(self, filename, data_matrix):
        filename = filename.upper()
        if 'RCS01A' in filename:
            return 0
        elif 'RCS01D' in filename:
            return 1
        elif 'RCS18A' in filename:
            return 2
        
        # 基于数据内容的启发式判断
        flow_mean = np.mean(data_matrix[:, 0:20])
        temp_mean = np.mean(data_matrix[:, 20:40])
        pressure_mean = np.mean(data_matrix[:, 40:60])
        
        if pressure_mean < 0.3 and flow_mean < 0.4:
            return 0
        elif temp_mean > 0.7 and flow_mean > 0.6:
            return 1
        else:
            return 2
    
    def load_all_data(self):
        all_files = [f for f in os.listdir(self.data_dir) if f.lower().endswith('.csv')]
        if not all_files:
            raise FileNotFoundError(f"No CSV files found in {self.data_dir}")
        
        data_list = []
        labels_list = []
        
        for file in all_files:
            filepath = os.path.join(self.data_dir, file)
            data_matrix, fault_type = self._load_and_validate_file(filepath)
            if data_matrix is not None:
                data_list.append(data_matrix)
                labels_list.extend([fault_type] * self.required_timesteps)
                print(f"Loaded {file} (Fault type: {fault_type})")
        
        if not data_list:
            raise RuntimeError("No valid files were loaded")
        
        return np.array(data_list), np.array(labels_list)

# ================== 信号处理器 ==================
class NuclearSignalProcessor:
    def __init__(self):
        self.param_groups = {
            'flow': slice(0, 20),
            'temperature': slice(20, 40),
            'pressure': slice(40, 60),
            'level': slice(60, 80),
            'other': slice(80, 88)
        }
    
    def normalize_signals(self, X):
        X_norm = np.zeros_like(X)
        for group, idx_slice in self.param_groups.items():
            group_data = X[:, :, idx_slice]
            min_vals = np.min(group_data, axis=(0, 1), keepdims=True)
            max_vals = np.max(group_data, axis=(0, 1), keepdims=True)
            range_vals = max_vals - min_vals
            range_vals[range_vals < 1e-8] = 1.0
            X_norm[:, :, idx_slice] = (group_data - min_vals) / range_vals
        return X_norm
    
    def smooth_signals(self, X, window=5, polyorder=2):
        X_smoothed = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[2]):
                try:
                    X_smoothed[i, :, j] = savgol_filter(X[i, :, j], window, polyorder)
                except:
                    X_smoothed[i, :, j] = X[i, :, j]
        return X_smoothed
    
    def reshape_to_images(self, X):
        n_samples, timesteps, n_features = X.shape
        total_images = n_samples * timesteps
        if n_features < 88:
            padded = np.zeros((n_samples, timesteps, 88))
            padded[:, :, :n_features] = X
            X = padded
        images = X.reshape(total_images, 88)
        return images.reshape(total_images, 11, 8, 1)
    
    def generate_fft_signals(self, X):
        """更稳定的FFT信号生成方法"""
        n_samples, timesteps, n_features = X.shape
        fft_signals = np.zeros((n_samples * timesteps, 1, 128))  # 减少到128点以降低维度
        
        for i in range(n_samples):
            for t in range(timesteps):
                signal = X[i, t, :]
                
                # 方法1: 标准FFT
                try:
                    # 补零到128点
                    padded_signal = np.pad(signal, (0, 128 - len(signal)), 'constant')
                    fft_result = np.fft.fft(padded_signal)
                    fft_magnitude = np.abs(fft_result)[:128]
                    log_psd = np.log1p(fft_magnitude)
                except Exception as e:
                    print(f"FFT processing error: {str(e)}")
                    # 如果失败，使用原始信号
                    log_psd = np.zeros(128)
                
                # 标准化
                log_psd = (log_psd - np.mean(log_psd)) / (np.std(log_psd) + 1e-8)
                fft_signals[i * timesteps + t, 0, :] = log_psd
                
        return fft_signals

# ================== 自注意力机制 ==================
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, input_dim, num_heads, output_dim):
        super().__init__()
        assert output_dim % num_heads == 0, "Output dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = output_dim // num_heads

        self.query = nn.Linear(input_dim, output_dim)
        self.key = nn.Linear(input_dim, output_dim)
        self.value = nn.Linear(input_dim, output_dim)
        self.fc_out = nn.Linear(output_dim, output_dim)

    def forward(self, x):
        B, L, D = x.shape
        Q = self.query(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attention_weights = F.softmax(attention_scores, dim=-1)
        weighted_values = torch.matmul(attention_weights, V)

        weighted_values = weighted_values.transpose(1, 2).contiguous().view(B, L, -1)
        return self.fc_out(weighted_values)

# ================== 核心模型 ==================
class ConvNetWithSelfAttention(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        
        # 2D-CNN部分 (处理图像数据)
        self.image_cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.AdaptiveAvgPool2d((2, 2)),  # 使用自适应池化确保输出尺寸一致
            nn.Dropout(0.3)
        )

        # 1D-CNN部分 (处理FFT信号)
        self.signal_cnn = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.1),
            nn.MaxPool1d(3),
            
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.MaxPool1d(3),
            
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.AdaptiveAvgPool1d(14),  # 确保输出长度一致
            nn.Dropout(0.3)
        )

        # 注意力机制
        self.attention1d = MultiHeadSelfAttention(128, 4, 128)
        
        # 特征融合层
        self.feature_fusion = nn.Sequential(
            nn.Linear(512 + 128*14, 512),  # 512 (图像特征) + 1792 (信号特征) = 2304
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3)
        )

        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(256, num_classes),
        )

    def forward(self, image_input, signal_input):
        # 处理图像数据
        image_features = self.image_cnn(image_input)
        image_features = image_features.view(image_features.size(0), -1)
        
        # 处理信号数据
        signal_features = self.signal_cnn(signal_input)
        signal_features = signal_features.permute(0, 2, 1)  # [batch, seq_len, features]
        signal_features = self.attention1d(signal_features)
        signal_features = signal_features.contiguous().view(signal_features.size(0), -1)
        
        # 特征融合
        combined_features = torch.cat([image_features, signal_features], dim=1)
        fused_features = self.feature_fusion(combined_features)
        
        # 分类
        logits = self.classifier(fused_features)
        probas = F.softmax(logits, dim=1)
        
        return logits, probas, image_features, signal_features, fused_features

# ================== 模型训练器 ==================
class NuclearModelTrainer:
    def __init__(self, model, train_loader, val_loader, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    def train(self, epochs=100, lr=0.001):
        # 使用标签平滑损失
        criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
        
        # 优化器配置
        optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=lr,
            weight_decay=1e-4
        )
        
        # 学习率调度器
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min',
            factor=0.5,
            patience=5
        )
        
        # 早停策略
        early_stop = EarlyStopping(patience=15, verbose=True)
        
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            correct = 0
            total = 0
            
            for x_img, x_sig, labels in self.train_loader:
                x_img, x_sig, labels = x_img.to(self.device), x_sig.to(self.device), labels.to(self.device)
                
                # 混合精度训练
                with torch.cuda.amp.autocast(enabled=(self.device.type == 'cuda')):
                    logits, _, _, _, _ = self.model(x_img, x_sig)
                    loss = criterion(logits, labels)
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(logits, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
            # 计算训练指标
            train_loss /= len(self.train_loader)
            train_acc = 100 * correct / total
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            
            # 验证阶段
            val_loss, val_acc = self.evaluate()
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            # 更新学习率
            scheduler.step(val_loss)
            
            # 打印进度
            print(f"Epoch {epoch+1}/{epochs}: "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # 早停检查
            early_stop(val_loss, self.model)
            if early_stop.early_stop:
                print("Early stopping")
                break
                
            # 释放内存
            gc.collect()
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
    
    def evaluate(self, loader=None):
        if loader is None:
            loader = self.val_loader
            
        self.model.eval()
        criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
        loss_val = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for x_img, x_sig, labels in loader:
                x_img, x_sig, labels = x_img.to(self.device), x_sig.to(self.device), labels.to(self.device)
                
                logits, probas, _, _, _ = self.model(x_img, x_sig)
                loss = criterion(logits, labels)
                
                loss_val += loss.item()
                _, predicted = torch.max(probas, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        loss_val /= len(loader)
        acc_val = 100 * correct / total
        return loss_val, acc_val
    
    def plot_training_history(self, save_path='training_history.png', dpi=600):
        history = self.history
        plt.figure(figsize=(10, 8))
        
        # 损失曲线
        plt.subplot(2, 1, 1)
        plt.plot(history['train_loss'], 'b-', label='Training Loss', linewidth=2)
        plt.plot(history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
        plt.title('Training and Validation Loss', fontsize=14, pad=10, weight='bold')
        plt.xlabel('Epochs', fontsize=12, weight='bold')
        plt.ylabel('Loss', fontsize=12, weight='bold')
        plt.legend(fontsize=11, framealpha=0.5)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tick_params(labelsize=11)
        
        # 准确率曲线
        plt.subplot(2, 1, 2)
        plt.plot(history['train_acc'], 'b-', label='Training Accuracy', linewidth=2)
        plt.plot(history['val_acc'], 'r-', label='Validation Accuracy', linewidth=2)
        plt.title('Training and Validation Accuracy', fontsize=14, pad=10, weight='bold')
        plt.xlabel('Epochs', fontsize=12, weight='bold')
        plt.ylabel('Accuracy (%)', fontsize=12, weight='bold')
        plt.legend(fontsize=11, framealpha=0.5)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tick_params(labelsize=11)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        plt.show()

    def plot_confusion_matrix(self, test_loader, save_path='confusion_matrix.png', dpi=600):
        """混淆矩阵可视化"""
        from sklearn.metrics import confusion_matrix, precision_score
        import seaborn as sns
        
        # 获取预测结果
        y_true = []
        y_pred = []
        
        self.model.eval()
        with torch.no_grad():
            for x_img, x_sig, labels in test_loader:
                x_img, x_sig = x_img.to(self.device), x_sig.to(self.device)
                _, probas, _, _, _ = self.model(x_img, x_sig)
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(torch.argmax(probas, dim=1).cpu().numpy())
        
        # 计算混淆矩阵
        cm = confusion_matrix(y_true, y_pred)
        precisions = precision_score(y_true, y_pred, average=None)
        fault_labels = ['Cold Leg', 'Hot Leg', 'Small Break']
        
        # 可视化
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=fault_labels, yticklabels=fault_labels)
        
        # 添加精度标签
        for i in range(len(precisions)):
            plt.text(i + 0.5, len(fault_labels) + 0.7, f'Precision: {precisions[i]:.2%}',
                    ha='center', va='center', fontsize=12, 
                    bbox=dict(boxstyle="round", alpha=0.2))
        
        plt.title('Confusion Matrix with Precision Scores', fontsize=14, pad=15, weight='bold')
        plt.xlabel('Predicted', fontsize=12, weight='bold')
        plt.ylabel('Actual', fontsize=12, weight='bold')
        plt.xticks(fontsize=11)
        plt.yticks(fontsize=11, rotation=0)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        plt.show()

    def visualize_tsne(self, test_loader, save_path='tsne_visualization.png', dpi=600):
        """3D t-SNE可视化"""
        from sklearn.manifold import TSNE
        from mpl_toolkits.mplot3d import Axes3D
        
        # 提取特征和标签
        features = []
        labels = []
        
        self.model.eval()
        with torch.no_grad():
            for x_img, x_sig, lbls in test_loader:
                x_img, x_sig = x_img.to(self.device), x_sig.to(self.device)
                _, _, _, _, fc_features = self.model(x_img, x_sig)
                features.append(fc_features.cpu().numpy())
                labels.append(lbls.cpu().numpy())
        
        features = np.concatenate(features)
        labels = np.concatenate(labels)
        
        # 应用3D t-SNE
        tsne = TSNE(n_components=3, random_state=42, perplexity=30)
        features_3d = tsne.fit_transform(features)
        
        # 可视化
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # 故障类型标签和颜色
        fault_labels = ['Cold Leg', 'Hot Leg', 'Small Break']
        colors = ['#FFB6C1', '#9370DB', '#90EE90']  # 浅粉、紫色、浅绿
        
        for i in range(3):  # 3个类别
            idx = np.where(labels == i)
            ax.scatter(features_3d[idx, 0], features_3d[idx, 1], features_3d[idx, 2],
                     c=colors[i], label=fault_labels[i], alpha=0.7, s=50,
                     edgecolor='k', linewidth=0.5)
        
        ax.set_xlabel('t-SNE 1', fontsize=13, weight='bold')
        ax.set_ylabel('t-SNE 2', fontsize=13, weight='bold')
        ax.set_zlabel('t-SNE 3', fontsize=13, weight='bold')
        ax.set_title('3D t-SNE Visualization of Fault Features', fontsize=14, pad=15, weight='bold')
        ax.legend(fontsize=12, framealpha=0.5)
        ax.grid(True, linestyle='--', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        plt.show()

    def plot_confusion_matrix_3d(self, test_loader, save_path='confusion_matrix_3d.png', dpi=600):
        """3D混淆矩阵可视化"""
        from sklearn.metrics import confusion_matrix, precision_score
        from mpl_toolkits.mplot3d import Axes3D
        
        # 获取预测结果
        y_true = []
        y_pred = []
        
        self.model.eval()
        with torch.no_grad():
            for x_img, x_sig, labels in test_loader:
                x_img, x_sig = x_img.to(self.device), x_sig.to(self.device)
                _, probas, _, _, _ = self.model(x_img, x_sig)
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(torch.argmax(probas, dim=1).cpu().numpy())
        
        # 计算混淆矩阵
        cm = confusion_matrix(y_true, y_pred)
        precisions = precision_score(y_true, y_pred, average=None)
        fault_labels = ['Cold Leg', 'Hot Leg', 'Small Break']
        colors = ['#FFB6C1', '#9370DB', '#90EE90']  # 浅粉、紫色、浅绿
        
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        xpos, ypos = np.meshgrid(np.arange(cm.shape[0]), np.arange(cm.shape[1]))
        xpos = xpos.flatten()
        ypos = ypos.flatten()
        zpos = np.zeros_like(xpos)
        dx = dy = 0.5 * np.ones_like(zpos)
        dz = cm.flatten()
        
        for i in range(len(xpos)):
            ax.bar3d(xpos[i], ypos[i], zpos[i], dx[i], dy[i], dz[i], 
                    color=colors[ypos[i]], alpha=0.8, edgecolor='k')
            
            if xpos[i] == ypos[i]:  # 对角线元素
                ax.text(xpos[i]+0.25, ypos[i]+0.25, dz[i]+0.5, 
                       f'{precisions[ypos[i]]:.2%}', 
                       color='k', fontsize=10, ha='center')
        
        ax.set_xticks(np.arange(cm.shape[0]) + 0.25)
        ax.set_yticks(np.arange(cm.shape[1]) + 0.25)
        ax.set_xticklabels(fault_labels, fontsize=10)
        ax.set_yticklabels(fault_labels, fontsize=10)
        ax.set_xlabel('Predicted', fontsize=11, labelpad=10)
        ax.set_ylabel('Actual', fontsize=11, labelpad=10)
        ax.set_zlabel('Count', fontsize=11, labelpad=10)
        ax.set_title('3D Confusion Matrix with Precision', fontsize=14, pad=15, weight='bold')
        ax.tick_params(axis='both', which='major', labelsize=9)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        plt.show()

# ================== 扩散模型辅助函数 ==================
def train_diffusion_model(X_flat, device, batch_size=64, num_epochs=500):
    """训练扩散模型用于数据增强"""
    print("\n[2/6] TRAINING DIFFUSION MODEL FOR DATA GENERATION")
    diffusion = DiffusionModel(input_dim=88).to(device)
    optimizer = torch.optim.Adam(diffusion.parameters(), lr=1e-3)
    
    # 创建数据集
    dataset = TensorDataset(X_flat)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    num_batches = len(loader)
    
    for epoch in range(num_epochs):
        diffusion.train()
        total_loss = 0.0
        
        for batch in loader:
            data = batch[0].to(device)
            
            # 随机采样时间步
            t = torch.randint(0, diffusion.timesteps, (data.size(0),), device=device)
            
            # 计算损失
            loss = diffusion.p_losses(data, t)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / num_batches
        if (epoch + 1) % 50 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
    
    print("Diffusion model training completed.")
    return diffusion

def generate_synthetic_data(diffusion, num_samples, device, batch_size=64):
    """使用扩散模型生成合成数据"""
    print("\n[3/6] GENERATING SYNTHETIC DATA")
    synthetic_batches = []
    
    # 分批次生成
    for _ in range(0, num_samples, batch_size):
        current_batch_size = min(batch_size, num_samples - len(synthetic_batches))
        synthetic_batch = diffusion.sample(
            batch_size=current_batch_size, 
            device=device
        )
        
        synthetic_batches.append(synthetic_batch)
    
    synthetic_data = torch.cat(synthetic_batches, dim=0)
    return synthetic_data

def evaluate_synthetic_data(diffusion, original_data, synthetic_data, key_params=None):
    """评估生成数据质量"""
    print("\n[4/6] EVALUATING GENERATED DATA QUALITY")
    if key_params is None:
        key_params = [0, 20, 40, 60, 80]  # 不同类型的关键参数
    
    evaluation_results = []
    
    for param_idx in key_params:
        result = diffusion.evaluate_generated_data(
            original_data[:, param_idx],
            synthetic_data[:len(original_data), param_idx],
            param_idx=param_idx
        )
        evaluation_results.append(result)
    
    # 打印总体评估
    mean_diffs = [res['mean_diff'] for res in evaluation_results]
    std_diffs = [res['std_diff'] for res in evaluation_results]
    correlations = [res['correlation'] for res in evaluation_results]
    mses = [res['mse'] for res in evaluation_results]
    
    print("\n===== Overall Synthetic Data Quality =====")
    print(f"Average Mean Difference: {np.mean(mean_diffs):.4f}")
    print(f"Average Std Difference: {np.mean(std_diffs):.4f}")
    print(f"Average MSE: {np.mean(mses):.6f}")
    print(f"Average Correlation: {np.mean(correlations):.4f}")
    
    # 可视化评估结果
    plt.figure(figsize=(14, 10))
    
    # 创建子图
    metrics = ['Mean', 'Std', 'Correlation', 'MSE']
    values = [mean_diffs, std_diffs, correlations, mses]
    
    for i, metric in enumerate(metrics):
        plt.subplot(2, 2, i+1)
        plt.bar(range(len(key_params)), values[i], color='#4F81BD', alpha=0.8)
        plt.axhline(y=np.mean(values[i]), color='r', linestyle='--', label='Average')
        plt.title(f'{metric} Comparison', fontsize=12)
        plt.xlabel('Parameter Index', fontsize=10)
        plt.ylabel(metric, fontsize=10)
        plt.xticks(range(len(key_params)), key_params)
        plt.grid(True, linestyle='--', alpha=0.3)
        if i == 2:
            plt.ylim(0, 1)  # 相关性范围
        plt.legend()
    
    plt.suptitle('Synthetic Data Quality Evaluation', fontsize=14, weight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('synthetic_data_quality_overview.png', dpi=300)
    plt.close()
    
    return evaluation_results

# ================== 主函数 ==================
def main():
    print("NUCLEAR POWER PLANT FAULT DIAGNOSIS SYSTEM WITH DATA AUGMENTATION")
    print("=" * 80)
    
    # 数据目录
    data_dir = "Data Sets"
    print(f"Data directory: {os.path.abspath(data_dir)}")
    
    try:
        # 1. 数据加载与处理
        print("\n[1/6] LOADING AND PROCESSING ORIGINAL DATA")
        loader = NuclearDataLoader(data_dir)
        X, y = loader.load_all_data()
        
        # 原始数据统计
        print(f"\nOriginal Data Statistics:")
        print(f"  Samples: {X.shape[0]}")
        print(f"  Timesteps per sample: {X.shape[1]}")
        print(f"  Parameters: {X.shape[2]}")
        print(f"  Fault type distribution: {np.bincount(y)}")
        
        # 将数据重塑为 (样本数*时间步数, 特征数) 用于扩散模型
        X_flat = X.reshape(-1, 88)
        X_flat_tensor = torch.tensor(X_flat, dtype=torch.float32)
        
        # 设备选择
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\nUsing device: {device}")
        
        # 2. 训练扩散模型
        diffusion = train_diffusion_model(X_flat_tensor, device)
        
        # 3. 生成合成数据
        num_synthetic_samples = 10 * len(X_flat_tensor)  # 10倍原始数据量
        synthetic_data = generate_synthetic_data(diffusion, num_synthetic_samples, device)
        
        # 4. 评估生成数据质量
        key_params = [0, 20, 40, 60, 80]  # 不同类型的关键参数
        evaluation_results = evaluate_synthetic_data(
            diffusion, 
            X_flat_tensor, 
            synthetic_data,
            key_params=key_params
        )
        
        # 5. 合并原始数据与合成数据
        print("\n[5/6] COMBINING ORIGINAL AND SYNTHETIC DATA")
        # 将原始数据重新组织为样本形式
        original_samples = X  # (30, 61, 88)
        
        # 将合成数据组织为样本形式 (每个样本61个时间步)
        synthetic_samples = synthetic_data.cpu().numpy().reshape(-1, 61, 88)
        
        # 为合成数据创建标签
        # 由于我们不知道合成数据的具体故障类型，使用原始数据的标签分布进行分配
        unique_labels, counts = np.unique(y, return_counts=True)
        label_dist = counts / counts.sum()
        synthetic_labels = np.random.choice(
            unique_labels, 
            size=len(synthetic_samples), 
            p=label_dist
        )
        
        # 合并数据
        combined_X = np.concatenate([original_samples, synthetic_samples], axis=0)
        combined_y = np.concatenate([y, synthetic_labels], axis=0)
        
        print(f"\nData Expansion Statistics:")
        print(f"  Original samples: {len(original_samples)}")
        print(f"  Synthetic samples: {len(synthetic_samples)}")
        print(f"  Combined samples: {len(combined_X)}")
        print(f"  Combined fault distribution: {np.bincount(combined_y)}")
        
        # 6. 使用扩展后的数据集继续处理
        print("\n[6/6] PROCESSING EXPANDED DATASET")
        processor = NuclearSignalProcessor()
        X_norm = processor.normalize_signals(combined_X)
        X_smooth = processor.smooth_signals(X_norm)
        
        # 生成2D图像输入
        X_images = processor.reshape_to_images(X_smooth)
        X_images = np.transpose(X_images, (0, 3, 1, 2))  # (N, C, H, W)
        
        # 生成1D FFT信号输入
        print("Generating FFT signals...")
        X_fft = processor.generate_fft_signals(X_smooth)
        print(f"FFT signals shape: {X_fft.shape}")
        
        # 转换为PyTorch张量
        y_tensor = torch.tensor(combined_y, dtype=torch.long)
        X_images_tensor = torch.tensor(X_images, dtype=torch.float32)
        X_fft_tensor = torch.tensor(X_fft, dtype=torch.float32)
        
        # 创建TensorDataset
        dataset = TensorDataset(X_images_tensor, X_fft_tensor, y_tensor)
        
        # 分割数据集
        train_val_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_val_size
        train_val_dataset, test_dataset = random_split(
            dataset, [train_val_size, test_size], 
            generator=torch.Generator().manual_seed(42)
        )
        
        train_size = int(0.85 * len(train_val_dataset))
        val_size = len(train_val_dataset) - train_size
        train_dataset, val_dataset = random_split(
            train_val_dataset, [train_size, val_size], 
            generator=torch.Generator().manual_seed(42)
        )
        
        # 创建数据加载器
        batch_size = 64  # 增加批处理大小
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # 7. 模型构建与训练
        print("\n[7/6] BUILDING AND TRAINING FAULT DIAGNOSIS MODEL")
        model = ConvNetWithSelfAttention(num_classes=3).to(device)
        print("\nModel architecture:")
        print(model)
        
        # 打印模型参数数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        trainer = NuclearModelTrainer(model, train_loader, val_loader, device)
        trainer.train(epochs=150, lr=0.001)  # 增加训练轮次
        
        # 8. 评估结果
        print("\n[8/6] EVALUATING RESULTS")
        # 使用测试集进行评估
        test_loss, test_acc = trainer.evaluate(test_loader)
        print(f"\nTest Accuracy: {test_acc:.2f}%") 
        print(f"Test Loss: {test_loss:.4f}")
        
        # 可视化训练历史和特征空间
        print("\nGenerating visualizations...")
        trainer.plot_training_history(save_path='training_history.png', dpi=600)
        trainer.visualize_tsne(test_loader, save_path='tsne_3d_visualization.png', dpi=600)
        trainer.plot_confusion_matrix_3d(test_loader, save_path='confusion_matrix_3d.png', dpi=600)
        
        # 保存模型
        torch.save(model.state_dict(), 'nuclear_fault_model.pth')
        print("\nModel saved as 'nuclear_fault_model.pth'")
        
        # 保存扩散模型
        torch.save(diffusion.state_dict(), 'diffusion_model.pth')
        print("Diffusion model saved as 'diffusion_model.pth'")

    except Exception as e:
        print(f"\nCRITICAL ERROR: {str(e)}")
        print("\nTROUBLESHOOTING TIPS:")
        print("1. Ensure 'Data Sets' folder exists with CSV files")
        print("2. Check CSV format - should contain numeric data")
        print("3. Verify file naming convention: RCS01A, RCS01D, RCS18A")
        print("4. Check available memory (dataset size: ~1830 samples)")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    start_time = time.time()
    main()
    print(f"\nTotal execution time: {time.time() - start_time:.2f} seconds")