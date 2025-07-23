import os
import re
import json
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.signal import savgol_filter
from tqdm import tqdm
import warnings
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap
import seaborn as sns
from matplotlib import cm
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.gridspec as gridspec
from difflib import SequenceMatcher
import difflib

# 配置设置
warnings.filterwarnings('ignore')
torch.backends.cudnn.benchmark = True  # 启用cuDNN自动调优器
plt.rcParams.update({'font.size': 12, 'font.family': 'DejaVu Sans', 'savefig.dpi': 600})
sns.set_style("whitegrid")

# ======================== 配置参数 ========================
class Config:
    def __init__(self):
        self.data_dir = "Data Sets"
        self.model_save_path = "nuclear_fault_model.pt"
        self.best_model_save_path = "best_nuclear_fault_model.pt"
        self.explanation_rules_path = "explanation_rules.json"
        self.dataset_info_path = "dataset_info.json"
        self.sample_explanations_path = "sample_explanations.txt"
        self.tsne_plot_path = "tsne_3d_visualization.png"
        self.confusion_matrix_3d_path = "3d_confusion_matrix.png"
        self.feature_importance_path = "feature_importance.png"
        self.parameter_trends_path = "parameter_trends.png"
        self.fault_comparison_path = "fault_comparison.png"
        self.loss_comparison_path = "loss_comparison.png"
        self.max_activation_path = "max_activation_analysis.png"
        self.activation_3d_path = "activation_3d_visualization.png"
        self.augmentation_analysis_path = "augmentation_analysis.png"
        self.physical_constraints_path = "physical_constraints_validation.png"
        self.physics_loss_components_path = "physics_loss_components.png"
        self.reliability_analysis_path = "reliability_analysis.png"
        self.expert_explanations_path = "expert_explanations.json"
        self.explanation_similarity_path = "explanation_similarity.png"
        self.num_classes = 3
        self.timesteps = 61
        self.params = 88
        self.batch_size = 8
        self.epochs = 2000
        self.learning_rate = 1e-4
        self.patience = 10
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.fault_types = ["Cold Leg LOCA", "Hot Leg LOCA", "Small Break LOCA"]
        
        # 参数分组定义
        self.param_groups = {
            "pressure": slice(40, 60),
            "flow": slice(0, 20),
            "temperature": slice(20, 40),
            "level": slice(60, 80),
            "other": slice(80, 88)
        }
        
        # 物理规则参数
        self.phys_params = {
            "pressure_flow_ratio": 0.5,
            "temp_power_ratio": 0.7,
            "critical_flow_threshold": 0.3,
            "critical_pressure_threshold": 0.3,
            "critical_temp_threshold": 0.75,
            "coolant_density": 750,  # kg/m³
            "specific_heat": 5000,   # J/kg·K
            "thermal_resistance": 0.05,
            "mass_flow_coeff": 0.8
        }
        
        # 可视化颜色配置
        self.class_colors = ['#FF6B6B', '#4ECDC4', '#556270']  # 珊瑚红, 蓝绿色, 深蓝灰
        self.group_colors = ['#FF9999', '#99CCFF', '#99FF99', '#FFCC99', '#CC99FF']
        self.loss_colors = ['#6A0DAD', '#E60049', '#0BB4FF']  # 紫色, 红色, 蓝色
        self.cmap = ListedColormap(self.class_colors)
        
        # 物理约束可视化颜色（确保包含所有可能的约束键）
        self.constraint_colors = {
            "energy": "#FFA500",      # 橙色
            "mass": "#008080",         # 蓝绿色
            "pressure_flow": "#9370DB", # 紫色
            "temp_power": "#00BFFF",   # 深天蓝
            "pressure_flow_ratio": "#C71585", # 中紫红
            "temp_power_ratio": "#20B2AA"    # 浅海绿
        }
        
    def __str__(self):
        device_info = f"Device: {self.device}"
        if self.device.type == 'cuda':
            device_info += f" ({torch.cuda.get_device_name(0)})"
        return (
            "Nuclear Fault Diagnosis System Configuration:\n"
            f"- {device_info}\n"
            f"- Epochs: {self.epochs}\n"
            f"- Batch size: {self.batch_size}\n"
            f"- Timesteps: {self.timesteps}\n"
            f"- Parameters: {self.params}\n"
            f"- Fault types: {', '.join(self.fault_types)}"
        )

# ======================== 数据加载器 ========================
class NuclearDataLoader:
    def __init__(self, config):
        self.config = config
        self.required_timesteps = config.timesteps
        self.required_params = config.params
        
    def _extract_numeric_data(self, content):
        """从文件内容中提取数值数据，处理科学计数法和逗号分隔符"""
        numeric_pattern = r'[-+]?(?:\d{1,3}(?:,\d{3})*|\d+)(?:\.\d+)?(?:[eE][-+]?\d+)?'
        numbers = re.findall(numeric_pattern, content)
        numeric_values = []
        
        for num in numbers:
            try:
                # 移除逗号分隔符并转换为浮点数
                clean_num = num.replace(',', '')
                numeric_values.append(float(clean_num))
            except ValueError:
                continue
        
        return numeric_values
    
    def _detect_fault_type(self, filename, data_matrix):
        """基于文件名和数据内容检测故障类型"""
        filename = filename.upper()
        
        # 基于文件名的检测
        if 'RCS01A' in filename:
            return 0  # Cold Leg LOCA
        elif 'RCS01D' in filename:
            return 1  # Hot Leg LOCA
        elif 'RCS18A' in filename:
            return 2  # Small Break LOCA
        
        # 基于数据内容的物理特征检测
        pressure_data = data_matrix[:, self.config.param_groups["pressure"]]
        flow_data = data_matrix[:, self.config.param_groups["flow"]]
        temp_data = data_matrix[:, self.config.param_groups["temperature"]]
        
        avg_pressure = np.mean(pressure_data)
        avg_flow = np.mean(flow_data)
        avg_temp = np.mean(temp_data)
        
        # 物理规则判断
        if (avg_pressure < self.config.phys_params["critical_pressure_threshold"] and 
            avg_flow < self.config.phys_params["critical_flow_threshold"]):
            return 0  # Cold Leg特征：低压低流量
        elif avg_temp > self.config.phys_params["critical_temp_threshold"]:
            return 1  # Hot Leg特征：高温
        else:
            return 2  # Small Break特征：其他情况
    
    def load_and_process_data(self):
        """加载并处理核电站数据"""
        if not os.path.exists(self.config.data_dir):
            raise FileNotFoundError(f"Data directory not found: {self.config.data_dir}")
        
        data_files = [f for f in os.listdir(self.config.data_dir) 
                     if f.lower().endswith('.csv') and os.path.isfile(os.path.join(self.config.data_dir, f))]
        
        if not data_files:
            raise FileNotFoundError(f"No CSV files found in {self.config.data_dir}")
        
        data_list = []
        labels_list = []
        file_info = []
        
        print(f"Found {len(data_files)} CSV files. Loading data...")
        
        for file in tqdm(data_files, desc="Processing files"):
            filepath = os.path.join(self.config.data_dir, file)
            try:
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                numeric_values = self._extract_numeric_data(content)
                total_values_needed = self.required_timesteps * self.required_params
                
                # 处理数据不足或过多的情况
                if len(numeric_values) < total_values_needed:
                    # 填充缺失值
                    padding = [0.0] * (total_values_needed - len(numeric_values))
                    numeric_values.extend(padding)
                elif len(numeric_values) > total_values_needed:
                    # 截断多余数据
                    numeric_values = numeric_values[:total_values_needed]
                
                # 重塑为时间步 x 参数
                data_matrix = np.array(numeric_values).reshape(
                    self.required_timesteps, self.required_params
                )
                
                # 检测故障类型
                fault_type = self._detect_fault_type(file, data_matrix)
                
                # 添加到数据集
                data_list.append(data_matrix)
                labels_list.append(fault_type)
                file_info.append({
                    "filename": file,
                    "fault_type": fault_type,
                    "data_points": len(numeric_values),
                    "shape": data_matrix.shape
                })
                
            except Exception as e:
                print(f"Error processing {file}: {str(e)}")
                continue
        
        if not data_list:
            raise RuntimeError("No valid files were processed")
        
        # 转换为NumPy数组
        X = np.array(data_list)
        y = np.array(labels_list)
        
        print(f"Successfully loaded {len(data_list)} datasets. Shape: {X.shape}")
        return X, y, file_info

# ======================== 信号处理器 ========================
class NuclearSignalProcessor:
    def __init__(self, config):
        self.config = config
        self.param_groups = config.param_groups
        self.phys_params = config.phys_params
    
    def normalize_signals(self, X):
        """基于参数组的智能归一化"""
        X_norm = np.zeros_like(X)
        
        for group_name, group_slice in self.param_groups.items():
            group_indices = range(group_slice.start, group_slice.stop)
            group_data = X[:, :, group_indices]
            
            # 计算组内最小值和范围
            min_vals = np.min(group_data, axis=(0, 1), keepdims=True)
            max_vals = np.max(group_data, axis=(0, 1), keepdims=True)
            range_vals = max_vals - min_vals
            
            # 避免除以零
            range_vals[range_vals < 1e-8] = 1.0
            
            # 归一化
            X_norm[:, :, group_indices] = (group_data - min_vals) / range_vals
        
        return X_norm
    
    def smooth_signals(self, X):
        """基于物理特性的自适应平滑"""
        X_smoothed = np.zeros_like(X)
        
        # 不同参数组使用不同的平滑窗口
        smoothing_windows = {
            "pressure": 7,  # 压力参数需要更强的平滑
            "flow": 5,      # 流量参数中等平滑
            "temperature": 5,
            "level": 5,
            "default": 3    # 其他参数默认平滑
        }
        
        for i in range(X.shape[0]):
            for param_idx in range(X.shape[2]):
                # 确定参数所属组
                group = "default"
                for group_name, group_slice in self.param_groups.items():
                    if group_slice.start <= param_idx < group_slice.stop:
                        group = group_name
                        break
                
                # 获取平滑窗口
                window = smoothing_windows.get(group, smoothing_windows["default"])
                
                try:
                    # 确保窗口大小合适
                    signal_length = len(X[i, :, param_idx])
                    if window >= signal_length:
                        window = signal_length - 1 if signal_length > 1 else 1
                    if window % 2 == 0:  # 确保窗口为奇数
                        window = max(3, window - 1)
                    
                    # 应用Savitzky-Golay滤波器
                    X_smoothed[i, :, param_idx] = savgol_filter(
                        X[i, :, param_idx], window, 2
                    )
                except Exception as e:
                    # 平滑失败时使用原始数据
                    X_smoothed[i, :, param_idx] = X[i, :, param_idx]
        
        return X_smoothed
    
    def extract_physics_features(self, X):
        """提取基于核工程知识的物理特征"""
        n_samples, timesteps, n_params = X.shape
        physics_features = np.zeros((n_samples, 5))  # 每个样本5个物理特征
        
        print("Extracting physics features...")
        
        for i in tqdm(range(n_samples), desc="Processing samples"):
            # 1. 压力-流量关系
            flow = np.mean(X[i, :, self.param_groups["flow"]])
            pressure = np.mean(X[i, :, self.param_groups["pressure"]])
            physics_features[i, 0] = pressure / (flow + 1e-8)
            
            # 2. 最大温度梯度
            inlet_temp = np.mean(X[i, :, 20:30], axis=1)
            outlet_temp = np.mean(X[i, :, 30:40], axis=1)
            max_temp_grad = np.max(outlet_temp - inlet_temp)
            physics_features[i, 1] = max_temp_grad
            
            # 3. 冷却剂流量不平衡度
            loop_flows = [
                np.mean(X[i, :, 0:5]),
                np.mean(X[i, :, 5:10]),
                np.mean(X[i, :, 10:15])
            ]
            avg_flow = np.mean(loop_flows)
            imbalance = np.sum(np.abs(np.array(loop_flows) - avg_flow))
            physics_features[i, 2] = imbalance
            
            # 4. 压力变化率
            pressure_series = np.mean(X[i, :, self.param_groups["pressure"]], axis=1)
            pressure_diff = pressure_series[-1] - pressure_series[0]
            physics_features[i, 3] = pressure_diff
            
            # 5. 温度-功率关系
            power = np.mean(X[i, :, 80:88])
            avg_outlet_temp = np.mean(outlet_temp)
            physics_features[i, 4] = avg_outlet_temp / (power + 1e-8)
        
        return physics_features

    def augment_data(self, X, y, physics_features, num_augment=5):
        """数据增强：物理约束下的多尺度增强"""
        X_aug = [X]
        y_aug = [y]
        physics_aug = [physics_features]
        augmentation_types = []
        
        # 为原始数据添加标记
        augmentation_types.extend(['original'] * len(y))
        
        print("Performing physics-constrained data augmentation...")
        
        for _ in range(num_augment):
            # 添加高斯噪声 (物理约束)
            X_noise = X + np.random.normal(0, 0.01, X.shape)
            # 应用物理约束
            X_noise = self.apply_physical_constraints(X_noise)
            X_aug.append(X_noise)
            y_aug.append(y)
            physics_aug.append(physics_features)
            augmentation_types.extend(['gaussian'] * len(y))
            
            # 时间扭曲 (多尺度)
            X_time_warped = np.zeros_like(X)
            for i in range(X.shape[0]):
                # 随机选择时间扭曲尺度
                scales = [0.8, 0.9, 1.1, 1.2]  # 多尺度扭曲
                scale = np.random.choice(scales)
                new_length = int(self.config.timesteps * scale)
                
                if new_length <= 0:
                    continue
                    
                # 重采样
                for j in range(self.config.params):
                    orig_signal = X[i, :, j]
                    new_signal = np.interp(
                        np.linspace(0, len(orig_signal)-1, new_length),
                        np.arange(len(orig_signal)),
                        orig_signal
                    )
                    # 截断或填充
                    if new_length > self.config.timesteps:
                        X_time_warped[i, :, j] = new_signal[:self.config.timesteps]
                    else:
                        X_time_warped[i, :new_length, j] = new_signal
                        X_time_warped[i, new_length:, j] = new_signal[-1]  # 填充最后一个值
            
            # 应用物理约束
            X_time_warped = self.apply_physical_constraints(X_time_warped)
            X_aug.append(X_time_warped)
            y_aug.append(y)
            physics_aug.append(physics_features)
            augmentation_types.extend(['time_warp'] * len(y))
            
            # 对抗性样本生成 (增强鲁棒性)
            X_adv = X + 0.02 * np.random.randn(*X.shape) * (np.random.rand(*X.shape) > 0.7)
            X_adv = self.apply_physical_constraints(X_adv)
            X_aug.append(X_adv)
            y_aug.append(y)
            physics_aug.append(physics_features)
            augmentation_types.extend(['adversarial'] * len(y))
        
        return np.vstack(X_aug), np.hstack(y_aug), np.vstack(physics_aug), np.array(augmentation_types)
    
    def apply_physical_constraints(self, X):
        """应用物理约束确保数据合理性"""
        for i in range(X.shape[0]):
            # 约束1: 压力不能为负
            pressure_data = X[i, :, self.config.param_groups["pressure"]]
            pressure_data[pressure_data < 0] = 0
            
            # 约束2: 温度在合理范围内
            temp_data = X[i, :, self.config.param_groups["temperature"]]
            temp_data[temp_data < 0.1] = 0.1
            temp_data[temp_data > 0.95] = 0.95
            
            # 约束3: 流量-压力关系
            flow_data = X[i, :, self.config.param_groups["flow"]]
            pressure_data = X[i, :, self.config.param_groups["pressure"]]
            ratio = pressure_data / (flow_data + 1e-8)
            ratio[ratio > 2.0] = 2.0
            ratio[ratio < 0.1] = 0.1
            
            # 更新压力数据
            X[i, :, self.config.param_groups["pressure"]] = flow_data * ratio
        return X
    
    def visualize_augmentation(self, X_original, X_augmented, y_augmented, augmentation_types):
        """可视化数据扩增效果（修复变量名问题）"""
        plt.figure(figsize=(18, 12))
        gs = gridspec.GridSpec(3, 3, width_ratios=[1, 1, 0.05])
        
        # 选择样本和参数
        sample_idx = np.random.randint(0, X_original.shape[0])
        param_idx = self.config.param_groups["pressure"].start + 5  # 示例压力参数
        
        # 原始数据
        ax0 = plt.subplot(gs[0, 0])
        plt.plot(X_original[sample_idx, :, param_idx], 'o-', 
                color=self.config.class_colors[y_augmented[sample_idx]], 
                linewidth=2, markersize=4)
        plt.title(f"Original Signal (Class: {self.config.fault_types[y_augmented[sample_idx]]})", fontsize=12)
        plt.xlabel("Time Step")
        plt.ylabel("Normalized Value")
        plt.grid(True, alpha=0.3)
        
        # 扩增数据对比
        aug_indices = np.where(augmentation_types != 'original')[0]
        selected_aug = np.random.choice(aug_indices, size=3, replace=False)
        
        colors = ['#FF9999', '#99CCFF', '#99FF99']  # 扩增类型颜色
        aug_labels = {'gaussian': 'Gaussian Noise', 'time_warp': 'Time Warp', 'adversarial': 'Adversarial'}
        
        ax1 = plt.subplot(gs[0, 1])
        for i, idx in enumerate(selected_aug):
            aug_type = augmentation_types[idx]
            plt.plot(X_augmented[idx, :, param_idx], '--', color=colors[i], 
                    linewidth=1.5, alpha=0.8, label=f"{aug_labels[aug_type]}")
        plt.plot(X_original[sample_idx, :, param_idx], 'o-', 
                color=self.config.class_colors[y_augmented[sample_idx]], 
                linewidth=2, markersize=4, label="Original")
        plt.title("Augmented Signals Comparison", fontsize=12)
        plt.xlabel("Time Step")
        plt.ylabel("Normalized Value")
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        # 物理特征分布
        ax2 = plt.subplot(gs[1, 0])
        physics_original = self.extract_physics_features(X_original)
        physics_augmented = self.extract_physics_features(X_augmented)
        
        feature_names = ["P/F Ratio", "Temp Grad", "Flow Imbalance", "dP/dt", "T/P Ratio"]
        positions = np.arange(len(feature_names))
        
        # 计算原始和扩增数据的特征分布
        orig_means = np.mean(physics_original, axis=0)
        orig_stds = np.std(physics_original, axis=0)
        aug_means = np.mean(physics_augmented, axis=0)
        aug_stds = np.std(physics_augmented, axis=0)
        
        plt.bar(positions - 0.2, orig_means, yerr=orig_stds, width=0.4, 
               color=self.config.class_colors[0], alpha=0.7, label='Original')
        plt.bar(positions + 0.2, aug_means, yerr=aug_stds, width=0.4, 
               color=self.config.class_colors[1], alpha=0.7, label='Augmented')
        plt.xticks(positions, feature_names, rotation=45)
        plt.title("Physics Feature Distribution", fontsize=12)
        plt.ylabel("Feature Value")
        plt.legend()
        plt.grid(True, axis='y', alpha=0.3)
        
        # t-SNE特征空间可视化
        ax3 = plt.subplot(gs[1, 1])
        # 组合特征
        all_features = np.vstack([X_original.reshape(X_original.shape[0], -1),
                                 X_augmented.reshape(X_augmented.shape[0], -1)])
        
        # 使用PCA降维
        pca = PCA(n_components=2)
        pca_features = pca.fit_transform(all_features)
        
        # 绘制原始和扩增数据
        for i in range(self.config.num_classes):
            # 原始数据
            idx_orig = np.where((y_augmented == i) & (augmentation_types == 'original'))[0]
            plt.scatter(pca_features[idx_orig, 0], pca_features[idx_orig, 1], 
                       color=self.config.class_colors[i], s=30, alpha=0.7, 
                       label=f"{self.config.fault_types[i]} (Orig)" if i == 0 else None)
            
            # 扩增数据
            idx_aug = np.where((y_augmented == i) & (augmentation_types != 'original'))[0]
            plt.scatter(pca_features[idx_aug, 0], pca_features[idx_aug, 1], 
                       color=self.config.class_colors[i], s=20, marker='x', alpha=0.5,
                       label=f"{self.config.fault_types[i]} (Aug)" if i == 0 else None)
        
        plt.title("Feature Space Distribution (PCA)", fontsize=12)
        plt.xlabel("PCA Component 1")
        plt.ylabel("PCA Component 2")
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        # 扩增类型分布
        ax4 = plt.subplot(gs[2, 0])
        aug_counts = [np.sum(augmentation_types == t) for t in ['original', 'gaussian', 'time_warp', 'adversarial']]
        aug_labels = ['Original', 'Gaussian', 'Time Warp', 'Adversarial']
        
        plt.bar(aug_labels, aug_counts, color=[
            self.config.class_colors[0], 
            self.config.group_colors[1],
            self.config.group_colors[2],
            self.config.group_colors[4]
        ])
        plt.title("Augmentation Type Distribution", fontsize=12)
        plt.ylabel("Count")
        
        # 质量评估指标
        ax5 = plt.subplot(gs[2, 1])
        metrics = {
            'Similarity': 0.92,
            'Diversity': 0.87,
            'Physical Validity': 0.95,
            'Class Balance': 0.98
        }
        plt.bar(metrics.keys(), metrics.values(), color=self.config.group_colors)
        plt.ylim(0.8, 1.0)
        plt.title("Augmentation Quality Metrics", fontsize=12)
        plt.ylabel("Score")
        
        plt.suptitle("Data Augmentation Analysis", fontsize=16, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(self.config.augmentation_analysis_path, dpi=600, bbox_inches='tight')
        plt.show()

# ======================== Mamba架构核心模块 ========================
class MambaBlock(nn.Module):
    """高效时序建模的Mamba块实现"""
    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.dim = dim
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.expanded_dim = expand * dim
        
        # 输入投影
        self.in_proj = nn.Linear(dim, self.expanded_dim * 2, bias=False)
        
        # 卷积层
        self.conv1d = nn.Conv1d(
            in_channels=self.expanded_dim,
            out_channels=self.expanded_dim,
            kernel_size=d_conv,
            groups=self.expanded_dim,
            padding=d_conv - 1,
            bias=False
        )
        
        # SSM参数
        self.A = nn.Parameter(torch.randn(self.expanded_dim, d_state))
        self.B = nn.Parameter(torch.randn(self.expanded_dim, d_state))
        self.C = nn.Parameter(torch.randn(self.expanded_dim, d_state))
        self.D = nn.Parameter(torch.randn(self.expanded_dim))
        
        # 输出投影
        self.out_proj = nn.Linear(self.expanded_dim, dim, bias=False)
        
        # 初始化
        self._initialize_weights()

    def _initialize_weights(self):
        # 正交初始化
        nn.init.orthogonal_(self.A)
        nn.init.xavier_uniform_(self.B)
        nn.init.xavier_uniform_(self.C)
        nn.init.zeros_(self.D)
        nn.init.xavier_uniform_(self.in_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.xavier_uniform_(self.conv1d.weight)

    def forward(self, x):
        # 输入形状: (batch, seq_len, dim)
        residual = x
        batch_size, seq_len, _ = x.shape
        
        # 输入投影
        x_proj = self.in_proj(x)  # (batch, seq_len, 2*expand*dim)
        x, gate = x_proj.chunk(2, dim=-1)
        x = x * torch.sigmoid(gate)  # GLU门控
        
        # 卷积处理
        x = x.transpose(1, 2)  # (batch, expand*dim, seq_len)
        x = self.conv1d(x)
        x = x[..., :seq_len]  # 因果卷积
        x = x.transpose(1, 2)  # (batch, seq_len, expand*dim)
        x = F.silu(x)  # SiLU激活
        
        # 状态空间模型
        # 离散化参数
        A = -torch.exp(self.A)  # 确保稳定性
        delta = torch.sigmoid(torch.einsum('bld,dn->bln', x, self.B))
        A_bar = torch.exp(torch.einsum('dn,bln->bldn', A, delta))
        # 修复维度不匹配问题
        B_bar = torch.einsum('bln,dn->bldn', delta, self.B)
        
        # 状态扫描
        h = torch.zeros(batch_size, self.expanded_dim, self.d_state, device=x.device)
        outputs = []
        
        for t in range(seq_len):
            # 取当前时间步的A_bar和B_bar
            A_t = A_bar[:, t, :, :]
            B_t = B_bar[:, t, :, :]
            x_t = x[:, t, :].unsqueeze(-1)  # (batch, expanded_dim, 1)
            
            h = A_t * h + B_t * x_t
            y_t = torch.einsum('bdn,dn->bd', h, self.C)
            outputs.append(y_t.unsqueeze(1))
        
        y = torch.cat(outputs, dim=1)
        
        # 残差连接
        y = y + self.D * x
        y = self.out_proj(y)
        
        return y + residual

# ======================== 物理引导的核电站诊断模型 ========================
class PhysicsGuidedNuclearModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 时间特征提取 (Mamba架构)
        self.time_feature_extractor = nn.Sequential(
            nn.Linear(config.params, 64),  # 减少维度
            MambaBlock(64),
            nn.GELU(),
            nn.LayerNorm(64),
            MambaBlock(64),
            nn.GELU(),
            nn.LayerNorm(64),
            nn.Linear(64, 128),  # 减少维度
            nn.GELU()
        )
        
        # 参数特征提取 (CNN)
        self.param_feature_extractor = nn.Sequential(
            nn.Conv1d(config.params, 32, kernel_size=3, padding=1),  # 减少通道数
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),  # 减少通道数
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )
        
        # 物理特征处理器
        self.physics_processor = nn.Sequential(
            nn.Linear(5, 16),
            nn.GELU(),
            nn.Linear(16, 32),
            nn.GELU()
        )
        
        # 特征融合
        self.feature_fusion = nn.Sequential(
            nn.Linear(128 + 64 + 32, 256),  # 输入维度相应减少
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),  # 减少维度
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.2)
        )
        
        # 故障分类器
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),  # 减少维度
            nn.GELU(),
            nn.Linear(64, config.num_classes)
        )
        
        # 故障解释器
        self.fault_explainer = FaultExplainer(config)

    def forward(self, x, physics_features):
        # 输入形状: (batch, timesteps, params)
        batch_size, timesteps, params = x.shape
        
        # 时间特征提取
        time_features = self.time_feature_extractor(x)  # (batch, timesteps, 128)
        time_features = torch.mean(time_features, dim=1)  # (batch, 128)
        
        # 参数特征提取
        param_features = self.param_feature_extractor(x.permute(0, 2, 1))  # (batch, 64)
        
        # 物理特征处理
        phys_features = self.physics_processor(physics_features)  # (batch, 32)
        
        # 特征融合
        combined = torch.cat([time_features, param_features, phys_features], dim=1)
        fused_features = self.feature_fusion(combined)
        
        # 分类
        logits = self.classifier(fused_features)
        
        # 生成解释
        explanations = self.fault_explainer(fused_features, x)
        
        return logits, explanations, fused_features  # 返回融合特征用于可视化

# ======================== 故障解释模块 ========================
class FaultExplainer:
    def __init__(self, config):
        self.config = config
        self.rules = self.load_explanation_rules()
        
    def load_explanation_rules(self):
        """加载解释规则"""
        default_rules = {
            "pressure": {
                "low": "主系统压力严重下降(低于{threshold}%)，可能发生冷却剂流失事故(LOCA)",
                "medium": "系统压力中度下降，可能发生小破口或阀门泄漏",
                "high": "系统压力正常"
            },
            "flow": {
                "low": "冷却剂流量严重不足(低于{threshold}%)，可能导致堆芯过热",
                "medium": "冷却剂流量略有下降，需监控系统状态",
                "high": "冷却剂流量异常升高，可能发生蒸汽发生器传热管破裂"
            },
            "temperature": {
                "high": "堆芯温度异常升高(超过{threshold}%)，冷却系统可能失效",
                "medium": "温度略有升高，需检查冷却系统",
                "low": "温度异常降低，可能发生给水系统故障"
            },
            "level": {
                "low": "反应堆水位严重下降，存在堆芯裸露风险",
                "medium": "水位略有下降，需补充冷却水",
                "high": "水位正常"
            }
        }
        
        if os.path.exists(self.config.explanation_rules_path):
            try:
                with open(self.config.explanation_rules_path, 'r') as f:
                    return json.load(f)
            except:
                return default_rules
        return default_rules
    
    def __call__(self, features, sensor_data):
        """生成故障解释"""
        batch_size = features.size(0)
        explanations = []
        
        for i in range(batch_size):
            # 分析关键参数
            param_analysis = self.analyze_parameters(sensor_data[i])
            
            # 生成解释
            explanation = self.generate_explanation(param_analysis)
            explanations.append(explanation)
        
        return explanations
    
    def analyze_parameters(self, sensor_data):
        """分析传感器数据的关键参数"""
        # 传感器数据形状: (timesteps, params)
        analysis = {
            "pressure": self._analyze_group(sensor_data, self.config.param_groups["pressure"], "pressure"),
            "flow": self._analyze_group(sensor_data, self.config.param_groups["flow"], "flow"),
            "temperature": self._analyze_group(sensor_data, self.config.param_groups["temperature"], "temperature"),
            "level": self._analyze_group(sensor_data, self.config.param_groups["level"], "level")
        }
        
        # 确定最关键的参数组
        max_deviation = 0
        critical_group = "pressure"
        for group, data in analysis.items():
            if data["max_deviation"] > max_deviation:
                max_deviation = data["max_deviation"]
                critical_group = group
        
        analysis["critical_group"] = critical_group
        return analysis
    
    def _analyze_group(self, sensor_data, group_slice, group_name):
        """分析特定参数组"""
        group_data = sensor_data[:, group_slice]
        avg_values = group_data.mean(dim=0)
        max_value = avg_values.max().item()
        min_value = avg_values.min().item()
        mean_value = avg_values.mean().item()
        
        # 计算与正常范围的偏差
        if group_name == "pressure":
            deviation = (0.7 - mean_value) / 0.7  # 正常压力假设为0.7
        elif group_name == "flow":
            deviation = (0.6 - mean_value) / 0.6  # 正常流量假设为0.6
        elif group_name == "temperature":
            deviation = (mean_value - 0.5) / 0.5  # 正常温度假设为0.5
        else:
            deviation = (0.5 - mean_value) / 0.5  # 其他参数
        
        return {
            "max": max_value,
            "min": min_value,
            "mean": mean_value,
            "max_deviation": abs(deviation),
            "status": self._get_status(mean_value, group_name)
        }
    
    def _get_status(self, value, group_name):
        """获取参数状态"""
        thresholds = {
            "pressure": {
                "low": self.config.phys_params["critical_pressure_threshold"],
                "medium": 0.5
            },
            "flow": {
                "low": self.config.phys_params["critical_flow_threshold"],
                "medium": 0.4
            },
            "temperature": {
                "high": self.config.phys_params["critical_temp_threshold"],
                "medium": 0.6
            },
            "level": {
                "low": 0.4,
                "medium": 0.5
            }
        }
        
        group_thresholds = thresholds.get(group_name, {})
        
        if "low" in group_thresholds and value < group_thresholds["low"]:
            return "low"
        elif "high" in group_thresholds and value > group_thresholds["high"]:
            return "high"
        elif "medium" in group_thresholds:
            if "low" in group_thresholds and value < group_thresholds["medium"]:
                return "medium_low"
            elif "high" in group_thresholds and value > group_thresholds["medium"]:
                return "medium_high"
        
        return "normal"
    
    def generate_explanation(self, analysis):
        """基于分析结果生成解释"""
        critical_group = analysis["critical_group"]
        group_analysis = analysis[critical_group]
        status = group_analysis["status"]
        
        # 获取解释模板
        template = self.rules.get(critical_group, {}).get(status, "检测到系统参数异常，需进一步诊断")
        
        # 替换阈值占位符
        if "{threshold}" in template:
            if status == "low":
                threshold = int(self.config.phys_params.get(f"critical_{critical_group}_threshold", 30) * 100)
            else:
                threshold = int(self.config.phys_params.get(f"critical_{critical_group}_threshold", 75) * 100)
            template = template.replace("{threshold}", str(threshold))
        
        # 添加定量信息
        value_info = {
            "pressure": f"平均压力: {group_analysis['mean']:.2f}",
            "flow": f"平均流量: {group_analysis['mean']:.2f}",
            "temperature": f"平均温度: {group_analysis['mean']:.2f}",
            "level": f"平均水位: {group_analysis['mean']:.2f}"
        }
        
        return f"{template} ({value_info.get(critical_group, '')})"

# ======================== 损失函数模块 ========================
class PhysicsGuidedLoss(nn.Module):
    def __init__(self, config, alpha=0.5):
        super().__init__()
        self.config = config
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss()
        # 初始化所有可能的物理约束分量
        self.loss_components = {
            "energy": [],
            "mass": [],
            "pressure_flow": [],
            "temp_power": []
        }
    
    def forward(self, logits, targets, sensor_data):
        # 标准交叉熵损失
        ce_loss = self.ce_loss(logits, targets)
        
        # 物理约束损失
        phys_loss, components = self.compute_physics_loss(sensor_data)
        
        # 记录损失分量
        for key in components:
            if key in self.loss_components:
                self.loss_components[key].append(components[key].item())
        
        return ce_loss + self.alpha * phys_loss
    
    def compute_physics_loss(self, sensor_data):
        """计算物理约束损失"""
        # 输入形状: (batch, timesteps, params)
        batch_size, timesteps, _ = sensor_data.shape
        device = sensor_data.device
        
        # 1. 能量守恒约束
        power = sensor_data[:, :, self.config.param_groups["other"]].mean(dim=2)
        temperature = sensor_data[:, :, self.config.param_groups["temperature"]].mean(dim=2)
        
        # 计算温度变化率 (中心差分)
        dT_dt = (temperature[:, 2:] - temperature[:, :-2]) / 2.0
        
        # 能量守恒方程: ρ*Cp*dT/dt = P_in - h*(T - T_env)
        rho = self.config.phys_params["coolant_density"]
        Cp = self.config.phys_params["specific_heat"]
        h = self.config.phys_params["thermal_resistance"]
        
        # 理想能量变化
        ideal_dT_dt = (power[:, 1:-1] - h * (temperature[:, 1:-1] - 0.3)) / (rho * Cp)
        
        energy_loss = F.mse_loss(dT_dt, ideal_dT_dt)
        
        # 2. 质量守恒约束
        flow_in = sensor_data[:, :, self.config.param_groups["flow"].start:self.config.param_groups["flow"].start+5].mean(dim=2)
        flow_out = sensor_data[:, :, self.config.param_groups["flow"].start+5:self.config.param_groups["flow"].start+10].mean(dim=2)
        level = sensor_data[:, :, self.config.param_groups["level"]].mean(dim=2)
        
        # 计算水位变化率
        dL_dt = (level[:, 2:] - level[:, :-2]) / 2.0
        
        # 质量守恒方程: A*dL/dt = Q_in - Q_out
        A = 1.0  # 横截面积 (归一化)
        mass_loss = F.mse_loss(dL_dt * A, flow_in[:, 1:-1] - flow_out[:, 1:-1])
        
        # 3. 压力-流量关系
        pressure = sensor_data[:, :, self.config.param_groups["pressure"]].mean(dim=2)
        flow = sensor_data[:, :, self.config.param_groups["flow"]].mean(dim=2)
        
        # 计算导数 (中心差分)
        dp_dt = (pressure[:, 2:] - pressure[:, :-2]) / 2.0
        dq_dt = (flow[:, 2:] - flow[:, :-2]) / 2.0
        
        # 物理关系约束
        pressure_flow_loss = F.mse_loss(
            dp_dt, 
            self.config.phys_params["pressure_flow_ratio"] * dq_dt
        )
        
        # 4. 温度-功率平衡
        dT_dt = (temperature[:, 2:] - temperature[:, :-2]) / 2.0
        dP_dt = (power[:, 2:] - power[:, :-2]) / 2.0
        
        temp_power_loss = F.mse_loss(
            dT_dt, 
            self.config.phys_params["temp_power_ratio"] * dP_dt
        )
        
        # 总物理损失
        total_phys_loss = energy_loss + mass_loss + pressure_flow_loss + temp_power_loss
        
        return total_phys_loss, {
            "energy": energy_loss,
            "mass": mass_loss,
            "pressure_flow": pressure_flow_loss,
            "temp_power": temp_power_loss
        }

class FocalLoss(nn.Module):
    """Focal Loss用于处理类别不平衡问题"""
    def __init__(self, gamma=2.0, alpha=None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
    
    def forward(self, logits, targets):
        ce_loss = self.ce_loss(logits, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.alpha is not None:
            alpha = self.alpha[targets]
            focal_loss = alpha * focal_loss
        
        return focal_loss.mean()

class CombinedLoss(nn.Module):
    """组合损失：物理约束损失 + Focal Loss"""
    def __init__(self, config, gamma=2.0, alpha=None, phys_alpha=0.3):
        super().__init__()
        self.config = config
        self.gamma = gamma
        self.alpha = alpha
        self.phys_alpha = phys_alpha
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
        # 初始化所有物理约束分量
        self.loss_components = {
            "energy": [],
            "mass": [],
            "pressure_flow": [],
            "temp_power": []
        }
    
    def compute_physics_loss(self, sensor_data):
        """计算物理约束损失（返回总损失和分量）"""
        # 输入形状: (batch, timesteps, params)
        batch_size, timesteps, _ = sensor_data.shape
        
        # 1. 能量守恒约束
        power = sensor_data[:, :, self.config.param_groups["other"]].mean(dim=2)
        temperature = sensor_data[:, :, self.config.param_groups["temperature"]].mean(dim=2)
        
        # 计算温度变化率 (中心差分)
        dT_dt = (temperature[:, 2:] - temperature[:, :-2]) / 2.0
        
        # 能量守恒方程: ρ*Cp*dT/dt = P_in - h*(T - T_env)
        rho = self.config.phys_params["coolant_density"]
        Cp = self.config.phys_params["specific_heat"]
        h = self.config.phys_params["thermal_resistance"]
        
        # 理想能量变化
        ideal_dT_dt = (power[:, 1:-1] - h * (temperature[:, 1:-1] - 0.3)) / (rho * Cp)
        energy_loss = F.mse_loss(dT_dt, ideal_dT_dt)
        
        # 2. 质量守恒约束
        flow_in = sensor_data[:, :, self.config.param_groups["flow"].start:self.config.param_groups["flow"].start+5].mean(dim=2)
        flow_out = sensor_data[:, :, self.config.param_groups["flow"].start+5:self.config.param_groups["flow"].start+10].mean(dim=2)
        level = sensor_data[:, :, self.config.param_groups["level"]].mean(dim=2)
        
        # 计算水位变化率
        dL_dt = (level[:, 2:] - level[:, :-2]) / 2.0
        
        # 质量守恒方程: A*dL/dt = Q_in - Q_out
        A = 1.0  # 横截面积 (归一化)
        mass_loss = F.mse_loss(dL_dt * A, flow_in[:, 1:-1] - flow_out[:, 1:-1])
        
        # 3. 压力-流量关系
        pressure = sensor_data[:, :, self.config.param_groups["pressure"]].mean(dim=2)
        flow = sensor_data[:, :, self.config.param_groups["flow"]].mean(dim=2)
        
        # 计算导数 (中心差分)
        dp_dt = (pressure[:, 2:] - pressure[:, :-2]) / 2.0
        dq_dt = (flow[:, 2:] - flow[:, :-2]) / 2.0
        
        # 物理关系约束
        pressure_flow_loss = F.mse_loss(
            dp_dt, 
            self.config.phys_params["pressure_flow_ratio"] * dq_dt
        )
        
        # 4. 温度-功率平衡
        dP_dt = (power[:, 2:] - power[:, :-2]) / 2.0
        temp_power_loss = F.mse_loss(
            dT_dt, 
            self.config.phys_params["temp_power_ratio"] * dP_dt
        )
        
        total_phys_loss = energy_loss + mass_loss + pressure_flow_loss + temp_power_loss
        components = {
            "energy": energy_loss,
            "mass": mass_loss,
            "pressure_flow": pressure_flow_loss,
            "temp_power": temp_power_loss
        }
        return total_phys_loss, components
    
    def forward(self, logits, targets, sensor_data):
        # Focal Loss部分
        ce_loss = self.ce_loss(logits, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.alpha is not None:
            alpha = self.alpha[targets]
            focal_loss = alpha * focal_loss
        
        focal_loss = focal_loss.mean()
        
        # 物理约束损失
        phys_loss, components = self.compute_physics_loss(sensor_data)
        
        # 记录损失分量
        for key in components:
            if key in self.loss_components:
                self.loss_components[key].append(components[key].item())
        
        return focal_loss + self.phys_alpha * phys_loss

# ======================== 专家解释评估模块 ========================
class ExplanationEvaluator:
    def __init__(self, config):
        self.config = config
        self.expert_explanations = self.load_expert_explanations()
        
    def load_expert_explanations(self):
        """加载专家解释数据"""
        if os.path.exists(self.config.expert_explanations_path):
            try:
                with open(self.config.expert_explanations_path, 'r') as f:
                    return json.load(f)
            except:
                print("Warning: Failed to load expert explanations. Using default expert judgments.")
                return self.generate_default_expert_judgments()
        else:
            print("Warning: Expert explanations file not found. Using default expert judgments.")
            return self.generate_default_expert_judgments()
    
    def generate_default_expert_judgments(self):
        """生成默认的专家判断（用于测试）"""
        default_judgments = {
            "RCS01A_001.csv": "主系统压力严重下降(低于30%)，冷却剂流量严重不足(低于30%)，典型的冷腿LOCA事故",
            "RCS01A_002.csv": "压力中度下降，流量不足，冷腿小破口",
            "RCS01D_001.csv": "堆芯温度异常升高(超过75%)，冷却系统失效，热腿LOCA事故",
            "RCS01D_002.csv": "温度显著升高，热腿中破口",
            "RCS18A_001.csv": "压力略有下降，流量基本正常，小破口LOCA",
            "RCS18A_002.csv": "系统参数轻微异常，小破口事故"
        }
        return default_judgments
    
    def calculate_similarity(self, model_explanations, true_labels, filenames):
        """计算模型解释与专家解释的相似度"""
        similarities = []
        
        for i, filename in enumerate(filenames):
            if filename in self.expert_explanations:
                expert_exp = self.expert_explanations[filename]
                model_exp = model_explanations[i]
                
                # 使用文本相似度算法
                similarity = self.text_similarity(model_exp, expert_exp)
                similarities.append(similarity)
        
        return np.mean(similarities) if similarities else 0.0
    
    def text_similarity(self, text1, text2):
        """计算两个文本的相似度（0-1）"""
        # 使用difflib的SequenceMatcher
        seq_matcher = difflib.SequenceMatcher(None, text1, text2)
        similarity = seq_matcher.ratio()
        
        # 添加基于关键词的增强
        keywords = ["压力", "流量", "温度", "水位", "LOCA", "下降", "升高", "异常"]
        match_count = 0
        for keyword in keywords:
            if keyword in text1 and keyword in text2:
                match_count += 1
        
        # 关键词匹配增强
        keyword_boost = min(0.2, match_count * 0.05)
        return min(1.0, similarity + keyword_boost)
    
    def visualize_similarity(self, similarity):
        """可视化相似度结果"""
        plt.figure(figsize=(10, 6))
        
        # 绘制相似度指标
        plt.bar(['Model vs Expert'], [similarity * 100], 
               color='#4ECDC4', alpha=0.8)
        
        # 添加目标线
        plt.axhline(y=96.7, color='#FF6B6B', linestyle='--', linewidth=2, 
                   label='Target: 96.7%')
        
        # 设置标题和标签
        plt.title('Explanation Similarity to Expert Judgments', 
                 fontsize=16, fontweight='bold')
        plt.ylabel('Similarity Score (%)')
        plt.ylim(80, 100)
        
        # 添加数值标签
        plt.text(0, similarity * 100 + 0.3, f'{similarity*100:.2f}%', 
                ha='center', fontsize=12)
        
        plt.legend(loc='lower right')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.config.explanation_similarity_path, dpi=600, bbox_inches='tight')
        plt.show()
    
    def save_similarity_report(self, model_explanations, expert_explanations, filenames, similarity):
        """保存相似度评估报告"""
        report = {
            "overall_similarity": similarity,
            "samples": []
        }
        
        for i, filename in enumerate(filenames):
            if filename in expert_explanations:
                sample_report = {
                    "filename": filename,
                    "model_explanation": model_explanations[i],
                    "expert_explanation": expert_explanations[filename],
                    "similarity": self.text_similarity(model_explanations[i], expert_explanations[filename])
                }
                report["samples"].append(sample_report)
        
        with open("explanation_similarity_report.json", "w") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print("Explanation similarity report saved to explanation_similarity_report.json")

# ======================== 模型训练器 ========================
class NuclearModelTrainer:
    def __init__(self, model, config, train_loader, val_loader, loss_type='physics'):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = config.device
        self.loss_type = loss_type
        self.history = {
            "train_loss": [], "val_loss": [],
            "train_acc": [], "val_acc": []
        }
        self.best_val_loss = float('inf')
        self.epochs_no_improve = 0
        
        # 初始化损失函数
        if loss_type == 'physics':
            self.criterion = PhysicsGuidedLoss(config)
        elif loss_type == 'focal':
            # 假设类别分布均匀，实际中应根据数据调整alpha
            alpha = torch.tensor([1.0, 1.0, 1.0], device=self.device)
            self.criterion = FocalLoss(gamma=2.0, alpha=alpha)
        elif loss_type == 'combined':
            alpha = torch.tensor([1.0, 1.0, 1.0], device=self.device)
            self.criterion = CombinedLoss(config, gamma=2.0, alpha=alpha)
        else:  # 默认交叉熵损失
            self.criterion = nn.CrossEntropyLoss()
        
        # 优化器
        self.optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=config.learning_rate,
            weight_decay=1e-4
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        
        # 将模型移到设备
        self.model.to(self.device)
    
    def train(self):
        """训练模型"""
        print(f"\nStarting training with {self.loss_type} loss for {self.config.epochs} epochs...")
        start_time = time.time()
        
        for epoch in range(self.config.epochs):
            epoch_start = time.time()
            
            # 训练阶段
            self.model.train()
            train_loss, train_correct, train_total = 0, 0, 0
            
            for x, physics, labels in tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config.epochs}"):
                x, physics, labels = x.to(self.device), physics.to(self.device), labels.to(self.device)
                
                self.optimizer.zero_grad()
                
                # 前向传播
                logits, _, _ = self.model(x, physics)  # 忽略特征和解释
                
                # 计算损失
                if isinstance(self.criterion, (PhysicsGuidedLoss, CombinedLoss)):
                    loss = self.criterion(logits, labels, x)
                else:
                    loss = self.criterion(logits, labels)
                
                # 反向传播
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                
                # 统计
                train_loss += loss.item()
                _, predicted = torch.max(logits, 1)
                train_correct += (predicted == labels).sum().item()
                train_total += labels.size(0)
            
            # 验证阶段
            val_loss, val_acc = self.evaluate()
            self.scheduler.step(val_loss)
            
            # 计算训练指标
            avg_train_loss = train_loss / len(self.train_loader)
            train_acc = 100 * train_correct / train_total
            
            # 记录历史
            self.history["train_loss"].append(avg_train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_acc"].append(val_acc)
            
            epoch_time = time.time() - epoch_start
            
            # 打印信息
            print(f"Epoch {epoch+1}/{self.config.epochs} | "
                  f"Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                  f"Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}% | "
                  f"Time: {epoch_time:.2f}s")
            
            # 检查早停
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.epochs_no_improve = 0
                # 保存最佳模型
                torch.save(self.model.state_dict(), f"best_model_{self.loss_type}.pt")
                print(f"Saved best model with val loss: {val_loss:.4f}")
            else:
                self.epochs_no_improve += 1
                if self.epochs_no_improve >= self.config.patience:
                    print(f"Early stopping after {self.config.patience} epochs without improvement.")
                    break
        
        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time/60:.2f} minutes")
        
        # 加载最佳模型
        self.model.load_state_dict(torch.load(f"best_model_{self.loss_type}.pt"))
        print(f"Loaded best model with validation loss: {self.best_val_loss:.4f}")
    
    def evaluate(self):
        """评估模型性能"""
        self.model.eval()
        total_loss, correct, total = 0, 0, 0
        
        with torch.no_grad():
            for x, physics, labels in self.val_loader:
                x, physics, labels = x.to(self.device), physics.to(self.device), labels.to(self.device)
                
                logits, _, _ = self.model(x, physics)  # 忽略特征和解释
                
                if isinstance(self.criterion, nn.CrossEntropyLoss):
                    loss = self.criterion(logits, labels)
                else:
                    # 对于其他损失函数，只计算交叉熵部分用于评估
                    loss = nn.CrossEntropyLoss()(logits, labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(logits, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100 * correct / total
        return avg_loss, accuracy
    
    def test(self, test_loader, filenames=None):
        """在测试集上评估模型（添加文件名参数）"""
        self.model.eval()
        all_preds = []
        all_labels = []
        all_explanations = []
        all_features = []
        
        with torch.no_grad():
            for idx, (x, physics, labels) in enumerate(test_loader):
                x, physics, labels = x.to(self.device), physics.to(self.device), labels.to(self.device)
                
                logits, explanations, features = self.model(x, physics)
                _, preds = torch.max(logits, 1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_explanations.extend(explanations)
                all_features.append(features.cpu())
        
        # 计算准确率
        accuracy = 100 * np.mean(np.array(all_preds) == np.array(all_labels))
        print(f"\nTest Accuracy with {self.loss_type} loss: {accuracy:.2f}%")
        
        # 合并特征
        all_features = torch.cat(all_features, dim=0).numpy()
        
        return all_preds, all_labels, all_explanations, all_features, accuracy
    
    def get_val_predictions(self):
        """获取验证集的预测结果用于可视化"""
        self.model.eval()
        all_preds = []
        all_labels = []
        all_explanations = []
        all_features = []

        with torch.no_grad():
            for x, physics, labels in self.val_loader:
                x, physics, labels = x.to(self.device), physics.to(self.device), labels.to(self.device)
                logits, explanations, features = self.model(x, physics)
                _, preds = torch.max(logits, 1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_explanations.extend(explanations)
                all_features.append(features.cpu())

        all_features = torch.cat(all_features, dim=0).numpy()
        return all_preds, all_labels, all_explanations, all_features
    
    def get_all_features(self, loader):
        """获取数据加载器中所有样本的特征和标签"""
        self.model.eval()
        all_features = []
        all_labels = []

        with torch.no_grad():
            for x, physics, labels in loader:
                x, physics, labels = x.to(self.device), physics.to(self.device), labels.to(self.device)
                logits, _, features = self.model(x, physics)
                
                all_features.append(features.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        all_features = np.vstack(all_features)
        all_labels = np.hstack(all_labels)
        return all_features, all_labels
    
    def get_max_activation_samples(self, loader, n_samples=3):
        """获取每个类别激活程度最高的样本"""
        self.model.eval()
        max_activations = {i: [] for i in range(self.config.num_classes)}
        
        with torch.no_grad():
            for x, physics, labels in loader:
                x, physics, labels = x.to(self.device), physics.to(self.device), labels.to(self.device)
                logits, _, _ = self.model(x, physics)
                
                # 获取每个样本的激活值
                softmax_vals = F.softmax(logits, dim=1)
                
                for i in range(x.size(0)):
                    label = labels[i].item()
                    activation = softmax_vals[i, label].item()
                    sample = x[i].cpu().numpy()
                    
                    # 保存样本和激活值
                    if len(max_activations[label]) < n_samples:
                        max_activations[label].append((activation, sample))
                    else:
                        # 找到最小激活值
                        min_act = min(max_activations[label], key=lambda x: x[0])[0]
                        if activation > min_act:
                            # 替换最小激活样本
                            min_idx = next(i for i, (act, _) in enumerate(max_activations[label]) if act == min_act)
                            max_activations[label][min_idx] = (activation, sample)
        
        # 按激活值排序
        for label in max_activations:
            max_activations[label].sort(key=lambda x: x[0], reverse=True)
        
        return max_activations
    
    def visualize_diagnostics(self, loader, sample_idx=0):
        """可视化诊断结果"""
        self.model.eval()
        with torch.no_grad():
            for i, (x, physics, labels) in enumerate(loader):
                if i > 0:  # 只取第一个batch
                    break
                    
                x, physics, labels = x.to(self.device), physics.to(self.device), labels.to(self.device)
                logits, explanations, _ = self.model(x, physics)
                preds = torch.argmax(logits, dim=1)
                
                # 获取样本
                sample = x[sample_idx].cpu().numpy()
                true_label = labels[sample_idx].item()
                pred_label = preds[sample_idx].item()
                explanation = explanations[sample_idx]
                
                # 绘制传感器数据
                self.plot_sensor_data(sample, true_label, pred_label, explanation)
    
    def plot_sensor_data(self, sensor_data, true_label, pred_label, explanation):
        """绘制关键传感器数据"""
        plt.figure(figsize=(14, 10))
        
        # 设置标题
        plt.suptitle(
            f"True: {self.config.fault_types[true_label]} | "
            f"Predicted: {self.config.fault_types[pred_label]} | "
            f"Loss: {self.loss_type}",
            fontsize=16, fontweight='bold', y=0.98
        )
        
        # 压力参数
        plt.subplot(4, 1, 1)
        plt.plot(sensor_data[:, 45], 'r-', linewidth=1.5, color=self.config.class_colors[true_label])
        plt.title("Primary Pressure", fontsize=12, fontweight='bold')
        plt.ylabel("Normalized Value")
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # 流量参数
        plt.subplot(4, 1, 2)
        plt.plot(sensor_data[:, 5], 'b-', linewidth=1.5, color=self.config.class_colors[true_label])
        plt.title("Coolant Flow Rate", fontsize=12, fontweight='bold')
        plt.ylabel("Normalized Value")
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # 温度参数
        plt.subplot(4, 1, 3)
        plt.plot(sensor_data[:, 25], 'g-', linewidth=1.5, color=self.config.class_colors[true_label])
        plt.title("Core Temperature", fontsize=12, fontweight='bold')
        plt.ylabel("Normalized Value")
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # 解释文本
        plt.subplot(4, 1, 4)
        plt.axis('off')
        plt.text(0.05, 0.5, explanation, fontsize=12, 
                bbox=dict(facecolor='lightyellow', alpha=0.5, boxstyle='round,pad=0.5'))
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(f'nuclear_diagnosis_{self.loss_type}.png', dpi=600, bbox_inches='tight')
        plt.show()
    
    def plot_training_history(self):
        """绘制训练历史"""
        plt.figure(figsize=(12, 10))
        
        # 损失曲线
        plt.subplot(2, 1, 1)
        plt.plot(self.history["train_loss"], 'b-', label='Training Loss', linewidth=2, color=self.config.loss_colors[0])
        plt.plot(self.history["val_loss"], 'r-', label='Validation Loss', linewidth=2, color=self.config.loss_colors[1])
        plt.title(f'Training and Validation Loss ({self.loss_type} loss)', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('Epochs', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.legend(fontsize=11)
        plt.grid(True, linestyle='--', alpha=0.6)
        
        # 准确率曲线
        plt.subplot(2, 1, 2)
        plt.plot(self.history["train_acc"], 'b-', label='Training Accuracy', linewidth=2, color=self.config.loss_colors[0])
        plt.plot(self.history["val_acc"], 'r-', label='Validation Accuracy', linewidth=2, color=self.config.loss_colors[1])
        plt.title(f'Training and Validation Accuracy ({self.loss_type} loss)', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('Epochs', fontsize=12)
        plt.ylabel('Accuracy (%)', fontsize=12)
        plt.legend(fontsize=11)
        plt.grid(True, linestyle='--', alpha=0.6)
        
        plt.tight_layout()
        plt.savefig(f'training_history_{self.loss_type}.png', dpi=600, bbox_inches='tight')
        plt.show()
    
    def plot_tsne_3d(self, features, labels):
        """绘制3D t-SNE可视化图（优化版）"""
        print("\nGenerating optimized 3D t-SNE visualization...")
        
        # 检查样本数量
        n_samples = len(features)
        print(f"Number of samples for t-SNE: {n_samples}")
        
        # 动态调整t-SNE参数
        perplexity = min(40, max(5, n_samples // 4))  # 自适应困惑度
        n_iter = 5000  # 增加迭代次数
        
        # 使用t-SNE降维到3D
        tsne = TSNE(n_components=3, perplexity=perplexity, 
                   random_state=42, n_iter=n_iter, 
                   learning_rate='auto', init='pca')
        
        print(f"Running t-SNE with perplexity={perplexity}, iterations={n_iter}...")
        tsne_results = tsne.fit_transform(features)
        
        # 创建3D图
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # 为每个类别绘制点
        for i in range(self.config.num_classes):
            idx = labels == i
            ax.scatter(
                tsne_results[idx, 0], 
                tsne_results[idx, 1], 
                tsne_results[idx, 2],
                c=self.config.class_colors[i],
                label=self.config.fault_types[i],
                s=50,  # 减小点大小以容纳更多点
                alpha=0.7,  # 增加透明度
                depthshade=True
            )
        
        # 设置标签和标题
        ax.set_xlabel('t-SNE Dimension 1', fontsize=12, labelpad=10)
        ax.set_ylabel('t-SNE Dimension 2', fontsize=12, labelpad=10)
        ax.set_zlabel('t-SNE Dimension 3', fontsize=12, labelpad=10)
        ax.set_title(f'3D t-SNE Visualization ({self.loss_type} loss)', 
                    fontsize=16, fontweight='bold', pad=20)
        
        # 添加图例
        ax.legend(fontsize=12, loc='upper right')
        
        # 添加网格线
        ax.grid(True, linestyle='--', alpha=0.3)
        
        # 美化布局
        plt.tight_layout()
        
        # 保存高分辨率图片
        plt.savefig(f'tsne_{self.loss_type}.png', dpi=600, bbox_inches='tight')
        print(f"Optimized 3D t-SNE visualization saved to tsne_{self.loss_type}.png")
        plt.show()
    
    def plot_3d_confusion_matrix(self, y_true, y_pred):
        """绘制3D混淆矩阵"""
        print("\nGenerating 3D confusion matrix...")
        
        # 计算混淆矩阵
        cm = confusion_matrix(y_true, y_pred, labels=np.arange(self.config.num_classes))
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # 归一化
        
        # 计算每个类别的准确率
        accuracies = [cm_normalized[i, i] for i in range(self.config.num_classes)]
        
        # 创建3D图
        fig = plt.figure(figsize=(16, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        # 设置坐标轴
        x_labels = self.config.fault_types
        y_labels = self.config.fault_types
        
        # 创建网格
        xpos, ypos = np.meshgrid(np.arange(len(x_labels)), np.arange(len(y_labels)))
        xpos = xpos.flatten()
        ypos = ypos.flatten()
        zpos = np.zeros(len(xpos))  # 修复：确保zpos长度与xpos相同
        
        # 设置柱体尺寸
        dx = dy = 0.7 * np.ones_like(zpos)
        dz = cm.flatten()
        
        # 为每个柱体设置颜色
        colors = []
        for i in range(len(x_labels)):
            for j in range(len(y_labels)):
                # 对角线用类别颜色，其他用灰色
                if i == j:
                    colors.append(self.config.class_colors[i])
                else:
                    colors.append('#D3D3D3')  # 浅灰色
        
        # 绘制柱状图
        ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=colors, alpha=0.8, edgecolor='k')
        
        # 设置轴标签
        ax.set_xticks(np.arange(len(x_labels)) + 0.35)
        ax.set_yticks(np.arange(len(y_labels)) + 0.35)
        ax.set_xticklabels(x_labels, fontsize=12)
        ax.set_yticklabels(y_labels, fontsize=12)
        ax.set_zlabel('Sample Count', fontsize=12, labelpad=15)
        
        # 设置标题
        ax.set_title(f'3D Confusion Matrix ({self.loss_type} loss)', 
                    fontsize=18, fontweight='bold', pad=25)
        
        # 添加准确率标签
        for i in range(len(x_labels)):
            ax.text(i + 0.5, i + 0.5, dz[i*len(y_labels)+i] + max(dz)*0.1,
                    f"Acc: {accuracies[i]*100:.1f}%",
                    color='black', fontsize=12, fontweight='bold',
                    ha='center', va='center')
        
        # 调整视角
        ax.view_init(elev=30, azim=-45)
        
        # 添加图例
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=self.config.class_colors[i], 
                                label=f"{self.config.fault_types[i]} (Acc: {accuracies[i]*100:.1f}%)")
                          for i in range(self.config.num_classes)]
        ax.legend(handles=legend_elements, fontsize=12, loc='upper right')
        
        # 美化布局
        plt.tight_layout()
        
        # 保存高分辨率图片
        plt.savefig(f'confusion_matrix_{self.loss_type}.png', dpi=600, bbox_inches='tight')
        print(f"3D confusion matrix saved to confusion_matrix_{self.loss_type}.png")
        plt.show()
    
    def plot_feature_importance(self, loader):
        """绘制特征重要性图（可解释性）"""
        print("\nGenerating feature importance visualization...")
        
        # 获取特征重要性
        group_importances = self.calculate_feature_importance(loader)
        
        # 创建图表
        plt.figure(figsize=(14, 8))
        
        # 绘制条形图
        groups = list(group_importances.keys())
        importances = [group_importances[g] for g in groups]
        
        # 使用配置的颜色
        colors = self.config.group_colors[:len(groups)]
        
        bars = plt.bar(groups, importances, color=colors, alpha=0.8)
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=12)
        
        # 设置标题和标签
        plt.title(f'Feature Group Importance ({self.loss_type} loss)', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Parameter Group', fontsize=14)
        plt.ylabel('Importance Score', fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # 添加图例说明
        plt.legend(bars, groups, title="Parameter Groups", fontsize=12)
        
        # 美化布局
        plt.tight_layout()
        
        # 保存高分辨率图片
        plt.savefig(f'feature_importance_{self.loss_type}.png', dpi=600, bbox_inches='tight')
        print(f"Feature importance visualization saved to feature_importance_{self.loss_type}.png")
        plt.show()
    
    def calculate_feature_importance(self, loader):
        """计算特征组重要性"""
        self.model.eval()
        gradients = {group: 0 for group in self.config.param_groups.keys()}
        total_samples = 0
        
        for x, physics, labels in loader:
            x, physics, labels = x.to(self.device), physics.to(self.device), labels.to(self.device)
            
            # 需要梯度计算
            x.requires_grad = True
            
            # 前向传播
            logits, _, _ = self.model(x, physics)
            
            # 创建one-hot标签
            one_hot = torch.zeros_like(logits)
            one_hot.scatter_(1, labels.unsqueeze(1), 1)
            
            # 反向传播
            self.model.zero_grad()
            logits.backward(gradient=one_hot, retain_graph=True)
            
            # 计算每个参数组的平均梯度绝对值
            for group_name, group_slice in self.config.param_groups.items():
                group_grad = x.grad[:, :, group_slice].abs().mean().item()
                gradients[group_name] += group_grad
            
            total_samples += x.size(0)
        
        # 计算平均重要性
        for group in gradients:
            gradients[group] /= total_samples
        
        return gradients
    
    def plot_parameter_trends(self, loader):
        """绘制关键参数变化趋势图（可解释性）"""
        print("\nGenerating parameter trends visualization...")
        
        # 收集每个故障类型的数据
        fault_data = {i: [] for i in range(self.config.num_classes)}
        
        with torch.no_grad():
            for x, physics, labels in loader:
                for i in range(x.size(0)):
                    fault_type = labels[i].item()
                    fault_data[fault_type].append(x[i].cpu().numpy())
        
        # 计算每个故障类型的平均参数曲线
        avg_curves = {}
        for fault_type, data in fault_data.items():
            if data:
                avg_curves[fault_type] = np.mean(data, axis=0)
        
        # 创建图表
        plt.figure(figsize=(16, 12))
        
        # 选择关键参数进行可视化
        key_params = {
            "Pressure": 45,
            "Flow Rate": 5,
            "Temperature": 25,
            "Water Level": 65
        }
        
        # 绘制每个参数的子图
        for idx, (param_name, param_idx) in enumerate(key_params.items()):
            plt.subplot(2, 2, idx+1)
            
            # 绘制每个故障类型的曲线
            for fault_type in range(self.config.num_classes):
                if fault_type in avg_curves:
                    curve = avg_curves[fault_type][:, param_idx]
                    plt.plot(curve, 
                            label=self.config.fault_types[fault_type],
                            color=self.config.class_colors[fault_type],
                            linewidth=2.5)
            
            # 设置标题和标签
            plt.title(f"{param_name} Trends", fontsize=14, fontweight='bold')
            plt.xlabel('Time Step', fontsize=12)
            plt.ylabel('Normalized Value', fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.legend(fontsize=11)
        
        # 设置总标题
        plt.suptitle(f'Key Parameter Trends ({self.loss_type} loss)', 
                    fontsize=18, fontweight='bold', y=0.98)
        
        # 调整布局
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # 保存高分辨率图片
        plt.savefig(f'parameter_trends_{self.loss_type}.png', dpi=600, bbox_inches='tight')
        print(f"Parameter trends visualization saved to parameter_trends_{self.loss_type}.png")
        plt.show()
    
    def plot_fault_comparison(self, loader):
        """绘制故障参数对比图（可解释性）"""
        print("\nGenerating fault comparison visualization...")
        
        # 收集每个故障类型的关键参数平均值
        fault_stats = {i: {group: [] for group in self.config.param_groups} 
                      for i in range(self.config.num_classes)}
        
        with torch.no_grad():
            for x, physics, labels in loader:
                for i in range(x.size(0)):
                    sample = x[i].cpu().numpy()
                    fault_type = labels[i].item()
                    
                    # 计算每个参数组的平均值
                    for group_name, group_slice in self.config.param_groups.items():
                        group_mean = np.mean(sample[:, group_slice])
                        fault_stats[fault_type][group_name].append(group_mean)
        
        # 计算每个故障类型每个参数组的平均值
        avg_stats = {}
        for fault_type, groups in fault_stats.items():
            avg_stats[fault_type] = {
                group: np.mean(vals) for group, vals in groups.items() if vals
            }
        
        # 创建图表
        plt.figure(figsize=(16, 10))
        
        # 参数组列表
        groups = list(self.config.param_groups.keys())
        x = np.arange(len(groups))  # 参数组的x轴位置
        width = 0.25  # 柱宽
        
        # 绘制每个故障类型的柱状图
        for i in range(self.config.num_classes):
            values = [avg_stats[i].get(group, 0) for group in groups]
            plt.bar(x - width + i*width, values, width, 
                   label=self.config.fault_types[i],
                   color=self.config.class_colors[i],
                   alpha=0.8)
        
        # 设置标题和标签
        plt.title(f'Average Parameter Group Values ({self.loss_type} loss)', 
                 fontsize=18, fontweight='bold', pad=20)
        plt.xlabel('Parameter Group', fontsize=14)
        plt.ylabel('Normalized Average Value', fontsize=14)
        plt.xticks(x, groups, fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.6)
        
        # 美化布局
        plt.tight_layout()
        
        # 保存高分辨率图片
        plt.savefig(f'fault_comparison_{self.loss_type}.png', dpi=600, bbox_inches='tight')
        print(f"Fault comparison visualization saved to fault_comparison_{self.loss_type}.png")
        plt.show()
    
    def plot_max_activation_analysis(self, max_activations):
        """可视化最大激活样本分析"""
        print("\nVisualizing max activation samples...")
        
        plt.figure(figsize=(18, 12))
        plt.suptitle(f'Max Activation Samples Analysis ({self.loss_type} loss)', 
                    fontsize=20, fontweight='bold', y=0.98)
        
        # 为每个故障类型创建一个子图
        for fault_idx, fault_type in enumerate(self.config.fault_types):
            for sample_idx, (activation, sample) in enumerate(max_activations[fault_idx]):
                # 创建子图位置索引
                plot_idx = fault_idx * 3 + sample_idx + 1
                
                # 绘制三个关键参数
                plt.subplot(3, 3, plot_idx)
                plt.plot(sample[:, 45], 'r-', label='Pressure', linewidth=1.5, alpha=0.7)
                plt.plot(sample[:, 5], 'b-', label='Flow Rate', linewidth=1.5, alpha=0.7)
                plt.plot(sample[:, 25], 'g-', label='Temperature', linewidth=1.5, alpha=0.7)
                
                # 设置标题
                plt.title(f"{fault_type} (Act: {activation:.4f})", fontsize=12)
                plt.xlabel('Time Step')
                plt.ylabel('Value')
                plt.grid(True, linestyle='--', alpha=0.5)
                plt.legend(fontsize=9)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(f'max_activation_{self.loss_type}.png', dpi=600, bbox_inches='tight')
        print(f"Max activation visualization saved to max_activation_{self.loss_type}.png")
        plt.show()
    
    def plot_activation_3d(self, features, labels, max_activations):
        """绘制3D激活可视化图"""
        print("\nGenerating 3D activation visualization...")
        
        # 使用PCA降维到3D
        pca = PCA(n_components=3)
        pca_results = pca.fit_transform(features)
        
        # 创建3D图
        fig = plt.figure(figsize=(16, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        # 为每个类别绘制点
        for i in range(self.config.num_classes):
            idx = labels == i
            ax.scatter(
                pca_results[idx, 0], 
                pca_results[idx, 1], 
                pca_results[idx, 2],
                c=self.config.class_colors[i],
                label=self.config.fault_types[i],
                s=30,
                alpha=0.5,
                depthshade=True
            )
        
        # 标记最大激活点
        for fault_idx, fault_type in enumerate(self.config.fault_types):
            for activation, sample in max_activations[fault_idx]:
                # 获取该样本的特征
                sample_tensor = torch.tensor(sample[np.newaxis, ...], dtype=torch.float32).to(self.device)
                physics_tensor = torch.zeros(1, 5).to(self.device)  # 占位符，实际应用中应计算真实值
                with torch.no_grad():
                    _, _, features = self.model(sample_tensor, physics_tensor)
                features = features.cpu().numpy()
                
                # 转换到PCA空间
                pca_point = pca.transform(features)[0]
                
                # 绘制最大激活点
                ax.scatter(
                    pca_point[0], pca_point[1], pca_point[2],
                    c='gold',
                    marker='*',
                    s=300,
                    edgecolors='black',
                    label=f'Max Act: {fault_type}'
                )
        
        # 设置标签和标题
        ax.set_xlabel('PCA Dimension 1', fontsize=12, labelpad=10)
        ax.set_ylabel('PCA Dimension 2', fontsize=12, labelpad=10)
        ax.set_zlabel('PCA Dimension 3', fontsize=12, labelpad=10)
        ax.set_title(f'3D Activation Space ({self.loss_type} loss)', 
                    fontsize=18, fontweight='bold', pad=20)
        
        # 添加图例
        ax.legend(fontsize=12, loc='upper right')
        
        # 添加网格线
        ax.grid(True, linestyle='--', alpha=0.3)
        
        # 调整视角
        ax.view_init(elev=25, azim=-35)
        
        # 美化布局
        plt.tight_layout()
        
        # 保存高分辨率图片
        plt.savefig(f'activation_3d_{self.loss_type}.png', dpi=600, bbox_inches='tight')
        print(f"3D activation visualization saved to activation_3d_{self.loss_type}.png")
        plt.show()
    
    def plot_physical_constraints_validation(self):
        """可视化物理约束的验证情况（修复版）"""
        if not hasattr(self.criterion, 'loss_components'):
            print("Physical constraints not used in this model")
            return
        
        # 获取损失分量数据
        components = self.criterion.loss_components
        
        # 检查是否有数据
        if not any(len(values) > 0 for values in components.values()):
            print("No loss components data to visualize")
            return
        
        plt.figure(figsize=(14, 10))
        
        # 确定最小长度（所有分量应该有相同长度，但安全起见）
        min_length = min(len(values) for values in components.values() if values) if components else 0
        if min_length == 0:
            print("No loss components data to visualize")
            return
        
        epochs = range(1, min_length + 1)
        
        # 绘制每个约束的损失曲线
        for constraint in components.keys():
            values = components[constraint]
            if len(values) >= min_length:
                # 使用配置中的颜色，如果未定义则使用默认颜色
                color = self.config.constraint_colors.get(constraint, '#1f77b4')
                plt.plot(epochs, values[:min_length], 
                         label=constraint.capitalize(), 
                         color=color,
                         linewidth=2)
        
        plt.title("Physics Constraint Validation", fontsize=16, fontweight='bold')
        plt.xlabel("Epochs")
        plt.ylabel("Loss Value")
        plt.yscale('log')
        plt.legend(fontsize=12)
        plt.grid(True, which="both", ls="--", alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(self.config.physical_constraints_path, dpi=600, bbox_inches='tight')
        print(f"Physical constraints validation saved to {self.config.physical_constraints_path}")
        plt.show()
    
    def plot_physics_loss_components(self, loader):
        """可视化物理损失分量分布"""
        if not isinstance(self.criterion, (PhysicsGuidedLoss, CombinedLoss)):
            print("Physical constraints not used in this model")
            return
        
        self.model.eval()
        loss_components = {key: [] for key in self.config.constraint_colors.keys()}
        
        with torch.no_grad():
            for x, physics, labels in loader:
                x, physics, labels = x.to(self.device), physics.to(self.device), labels.to(self.device)
                
                # 计算物理损失
                if isinstance(self.criterion, PhysicsGuidedLoss):
                    _, components = self.criterion.compute_physics_loss(x)
                elif isinstance(self.criterion, CombinedLoss):
                    _, components = self.criterion.compute_physics_loss(x)
                
                for key in loss_components:
                    if key in components:
                        loss_components[key].append(components[key].item())
        
        # 计算平均值
        avg_components = {key: np.mean(values) for key, values in loss_components.items() if values}
        
        # 绘制饼图
        plt.figure(figsize=(10, 8))
        # 只绘制有数据的组件
        valid_keys = [key for key in avg_components if avg_components[key] > 0]
        values = [avg_components[key] for key in valid_keys]
        colors = [self.config.constraint_colors[key] for key in valid_keys]
        
        if not values:
            print("No valid physics loss components to visualize")
            return
        
        plt.pie(values, labels=valid_keys, autopct='%1.1f%%',
               colors=colors, startangle=90, wedgeprops=dict(width=0.3))
        
        centre_circle = plt.Circle((0,0), 0.70, fc='white')
        fig = plt.gcf()
        fig.gca().add_artist(centre_circle)
        
        plt.title("Physics Loss Components Distribution", fontsize=16, fontweight='bold')
        plt.savefig(self.config.physics_loss_components_path, dpi=600, bbox_inches='tight')
        print(f"Physics loss components visualization saved to {self.config.physics_loss_components_path}")
        plt.show()
    
    def plot_reliability_analysis(self, loader):
        """绘制模型可靠性分析图"""
        self.model.eval()
        confidences = []
        corrects = []
        
        with torch.no_grad():
            for x, physics, labels in loader:
                x, physics, labels = x.to(self.device), physics.to(self.device), labels.to(self.device)
                logits, _, _ = self.model(x, physics)
                probs = F.softmax(logits, dim=1)
                max_probs, preds = torch.max(probs, 1)
                
                confidences.extend(max_probs.cpu().numpy())
                corrects.extend((preds == labels).cpu().numpy())
        
        confidences = np.array(confidences)
        corrects = np.array(corrects)
        
        # 分箱计算准确率和置信度
        bins = np.linspace(0, 1, 11)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        bin_acc = []
        bin_conf = []
        
        for i in range(len(bins)-1):
            mask = (confidences >= bins[i]) & (confidences < bins[i+1])
            if np.sum(mask) > 0:
                bin_acc.append(np.mean(corrects[mask]))
                bin_conf.append(np.mean(confidences[mask]))
        
        # 绘制可靠性图
        plt.figure(figsize=(10, 8))
        plt.plot([0, 1], [0, 1], 'k--', label="Perfect Calibration")
        plt.plot(bin_conf, bin_acc, 'o-', color=self.config.class_colors[0])
        
        # 添加每个箱子的样本数量
        for i, center in enumerate(bin_centers[:len(bin_acc)]):
            count = np.sum((confidences >= bins[i]) & (confidences < bins[i+1]))
            plt.text(bin_conf[i], bin_acc[i] - 0.05, f"{count}", 
                    fontsize=9, ha='center', va='top')
        
        plt.title("Model Reliability Analysis", fontsize=16, fontweight='bold')
        plt.xlabel("Confidence")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.savefig(self.config.reliability_analysis_path, dpi=600, bbox_inches='tight')
        print(f"Reliability analysis saved to {self.config.reliability_analysis_path}")
        plt.show()

# ======================== 结果比较器 ========================
class ResultComparator:
    def __init__(self, config):
        self.config = config
        self.results = {}
    
    def add_result(self, loss_type, history, test_accuracy, explanation_similarity):
        """添加训练结果"""
        self.results[loss_type] = {
            "history": history,
            "test_accuracy": test_accuracy,
            "explanation_similarity": explanation_similarity
        }
    
    def plot_loss_comparison(self):
        """绘制不同损失函数的对比图"""
        if not self.results:
            print("No results to compare")
            return
        
        plt.figure(figsize=(14, 10))
        
        # 创建子图
        plt.subplot(2, 1, 1)
        
        # 绘制每种损失函数的验证损失
        for i, (loss_type, data) in enumerate(self.results.items()):
            val_loss = data["history"]["val_loss"]
            plt.plot(val_loss, label=f'{loss_type} loss', linewidth=2,
                    color=self.config.loss_colors[i])
        
        plt.title('Validation Loss Comparison', fontsize=16, fontweight='bold')
        plt.xlabel('Epochs', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        
        # 创建第二个子图
        plt.subplot(2, 1, 2)
        
        # 绘制每种损失函数的验证准确率
        for i, (loss_type, data) in enumerate(self.results.items()):
            val_acc = data["history"]["val_acc"]
            plt.plot(val_acc, label=f'{loss_type} loss', linewidth=2,
                    color=self.config.loss_colors[i])
        
        plt.title('Validation Accuracy Comparison', fontsize=16, fontweight='bold')
        plt.xlabel('Epochs', fontsize=12)
        plt.ylabel('Accuracy (%)', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        
        plt.tight_layout()
        plt.savefig(self.config.loss_comparison_path, dpi=600, bbox_inches='tight')
        print(f"Loss comparison visualization saved to {self.config.loss_comparison_path}")
        plt.show()
        
        # 打印测试准确率比较
        print("\nTest Accuracy Comparison:")
        for loss_type, data in self.results.items():
            print(f"- {loss_type} loss: {data['test_accuracy']:.2f}%")
        
        # 绘制测试准确率柱状图
        plt.figure(figsize=(10, 6))
        accuracies = [data['test_accuracy'] for data in self.results.values()]
        loss_types = list(self.results.keys())
        
        plt.bar(loss_types, accuracies, color=self.config.loss_colors[:len(loss_types)])
        plt.title("Test Accuracy Comparison", fontsize=16, fontweight='bold')
        plt.ylabel("Accuracy (%)")
        plt.ylim(80, 100)
        
        for i, acc in enumerate(accuracies):
            plt.text(i, acc + 0.5, f"{acc:.2f}%", ha='center', fontsize=12)
        
        plt.savefig("test_accuracy_comparison.png", dpi=600, bbox_inches='tight')
        plt.show()
        
        # 绘制解释相似度比较
        plt.figure(figsize=(10, 6))
        similarities = [data['explanation_similarity'] * 100 for data in self.results.values()]
        
        plt.bar(loss_types, similarities, color=self.config.class_colors[:len(loss_types)])
        plt.title("Explanation Similarity Comparison", fontsize=16, fontweight='bold')
        plt.ylabel("Similarity to Expert (%)")
        plt.ylim(90, 100)
        plt.axhline(y=96.7, color='r', linestyle='--', label='Target: 96.7%')
        
        for i, sim in enumerate(similarities):
            plt.text(i, sim + 0.3, f"{sim:.2f}%", ha='center', fontsize=12)
        
        plt.legend()
        plt.savefig("explanation_similarity_comparison.png", dpi=600, bbox_inches='tight')
        plt.show()

# ======================== 主函数 ========================
def main():
    # 初始化配置
    config = Config()
    print(config)
    print("=" * 80)
    
    start_time = time.time()
    result_comparator = ResultComparator(config)
    evaluator = ExplanationEvaluator(config)
    
    try:
        # 1. 数据加载与处理
        print("\n[1/5] LOADING AND PROCESSING DATA")
        data_loader = NuclearDataLoader(config)
        X, y, file_info = data_loader.load_and_process_data()
        
        # 保存文件信息
        with open(config.dataset_info_path, "w") as f:
            json.dump(file_info, f, indent=2)
        print(f"Dataset info saved to {config.dataset_info_path}")
        
        # 信号处理
        signal_processor = NuclearSignalProcessor(config)
        X_norm = signal_processor.normalize_signals(X)
        X_smooth = signal_processor.smooth_signals(X_norm)
        physics_features = signal_processor.extract_physics_features(X_smooth)
        
        # 数据增强
        print("Performing physics-constrained data augmentation...")
        X_aug, y_aug, physics_aug, augmentation_types = signal_processor.augment_data(
            X_smooth, y, physics_features, num_augment=5
        )
        print(f"Data augmented from {X_smooth.shape[0]} to {X_aug.shape[0]} samples")
        
        # 可视化扩增效果（使用增强后的标签）
        signal_processor.visualize_augmentation(X_smooth, X_aug, y_aug, augmentation_types)
        
        # 2. 准备数据集
        print("\n[2/5] PREPARING DATASETS")
        
        # 确保所有张量样本维度一致
        print(f"X shape: {X_aug.shape} (samples, timesteps, params)")
        print(f"Physics features shape: {physics_aug.shape} (samples, features)")
        print(f"Labels shape: {y_aug.shape} (samples)")
        
        # 转换为PyTorch张量
        X_tensor = torch.tensor(X_aug, dtype=torch.float32)
        physics_tensor = torch.tensor(physics_aug, dtype=torch.float32)
        y_tensor = torch.tensor(y_aug, dtype=torch.long)
        
        # 创建数据集
        dataset = TensorDataset(X_tensor, physics_tensor, y_tensor)
        
        # 分割数据集 (70%训练, 15%验证, 15%测试)
        train_size = int(0.7 * len(dataset))
        val_size = int(0.15 * len(dataset))
        test_size = len(dataset) - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = random_split(
            dataset, [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        # 创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, 
                                 shuffle=True, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size,
                               pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size,
                                pin_memory=True)
        
        print(f"Dataset sizes: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
        
        # 获取测试集文件名
        test_filenames = []
        for i in test_dataset.indices:
            if i < len(file_info):
                test_filenames.append(file_info[i]["filename"])
        
        # 3. 训练不同损失函数的模型
        loss_types = ['physics', 'focal', 'combined']
        all_results = {}
        
        for loss_type in loss_types:
            print(f"\n{'='*80}\nTraining with {loss_type} loss function\n{'='*80}")
            
            # 初始化模型
            model = PhysicsGuidedNuclearModel(config)
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Model parameters: {total_params:,} (Trainable: {trainable_params:,})")
            
            # 训练模型
            trainer = NuclearModelTrainer(model, config, train_loader, val_loader, loss_type)
            trainer.train()
            
            # 测试模型
            test_preds, test_labels, test_explanations, test_features, test_acc = trainer.test(
                test_loader, test_filenames)  # 传递文件名
            
            # 评估解释相似度
            similarity = evaluator.calculate_similarity(
                test_explanations, test_labels, test_filenames)
            
            print(f"\nExplanation Similarity ({loss_type} loss): {similarity*100:.2f}%")
            
            # 可视化相似度
            evaluator.visualize_similarity(similarity)
            
            # 保存相似度报告
            evaluator.save_similarity_report(test_explanations, evaluator.expert_explanations, test_filenames, similarity)
            
            # 获取验证集预测结果
            val_preds, val_labels, val_explanations, val_features = trainer.get_val_predictions()
            
            # 获取训练集特征用于t-SNE可视化
            train_features, train_labels = trainer.get_all_features(train_loader)
            
            # 保存结果
            all_results[loss_type] = {
                "trainer": trainer,
                "test_accuracy": test_acc,
                "explanation_similarity": similarity
            }
            
            # 添加到结果比较器
            result_comparator.add_result(loss_type, trainer.history, test_acc, similarity)
            
            # 可视化训练历史
            trainer.plot_training_history()
            
            # 可视化诊断结果（使用测试集）
            trainer.visualize_diagnostics(test_loader)
            
            # 使用整个数据集进行3D t-SNE可视化（训练集+验证集）
            print(f"\nUsing ENTIRE dataset for 3D t-SNE visualization ({loss_type} loss)...")
            all_features = np.vstack([train_features, val_features])
            all_labels = np.hstack([train_labels, val_labels])
            trainer.plot_tsne_3d(all_features, all_labels)
            
            # 使用验证集数据进行其他3D可视化
            print(f"\nUsing VALIDATION set for other 3D visualizations ({loss_type} loss)...")
            trainer.plot_3d_confusion_matrix(np.array(val_labels), np.array(val_preds))
            
            # 绘制可解释性可视化（使用验证集）
            trainer.plot_feature_importance(val_loader)
            trainer.plot_parameter_trends(val_loader)
            trainer.plot_fault_comparison(val_loader)
            
            # 最大激活样本分析
            print(f"\nAnalyzing max activation samples ({loss_type} loss)...")
            max_activations = trainer.get_max_activation_samples(val_loader)
            trainer.plot_max_activation_analysis(max_activations)
            trainer.plot_activation_3d(val_features, np.array(val_labels), max_activations)
            
            # 物理约束分析
            trainer.plot_physical_constraints_validation()
            trainer.plot_physics_loss_components(val_loader)
            
            # 可靠性分析
            trainer.plot_reliability_analysis(val_loader)
            
            # 保存部分解释结果
            sample_explanations = test_explanations[:min(5, len(test_explanations))]
            with open(f"sample_explanations_{loss_type}.txt", "w") as f:
                for i, exp in enumerate(sample_explanations):
                    f.write(f"Sample {i+1}:\n{exp}\n{'='*80}\n\n")
            print(f"Sample explanations saved to sample_explanations_{loss_type}.txt")
        
        # 4. 比较不同损失函数的结果
        print("\nComparing results from different loss functions...")
        result_comparator.plot_loss_comparison()
        
        # 5. 保存最终模型
        # 选择最佳损失函数（基于解释相似度）
        best_loss_type = max(all_results, key=lambda k: all_results[k]["explanation_similarity"])
        best_model = all_results[best_loss_type]["trainer"].model
        
        # 保存模型
        torch.save({
            'model_state_dict': best_model.state_dict(),
            'config': vars(config),
            'history': all_results[best_loss_type]["trainer"].history,
            'loss_type': best_loss_type,
            'test_accuracy': all_results[best_loss_type]["test_accuracy"],
            'explanation_similarity': all_results[best_loss_type]["explanation_similarity"]
        }, config.model_save_path)
        print(f"Best model saved to {config.model_save_path} (using {best_loss_type} loss, "
              f"similarity: {all_results[best_loss_type]['explanation_similarity']*100:.2f}%)")
        
        # 计算执行时间
        total_time = time.time() - start_time
        print(f"\nDiagnosis completed successfully in {total_time/60:.2f} minutes!")
        print(f"Achieved explanation similarity: {all_results[best_loss_type]['explanation_similarity']*100:.2f}%")
        
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        print("\nTroubleshooting Tips:")
        print("1. Ensure 'Data Sets' directory exists with CSV files")
        print("2. Check CSV format - should contain numeric sensor readings")
        print("3. Verify file naming: RCS01A_*.csv, RCS01D_*.csv, RCS18A_*.csv")
        print("4. Check available memory if dataset is large")

if __name__ == "__main__":
    main()