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
from scipy.signal import savgol_filter
from tqdm import tqdm
import warnings
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap
import seaborn as sns
from matplotlib import cm

# 配置设置
warnings.filterwarnings('ignore')
torch.backends.cudnn.benchmark = True  # 启用cuDNN自动调优器

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
        self.num_classes = 3
        self.timesteps = 61
        self.params = 88
        self.batch_size = 8
        self.epochs = 500
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
            "critical_temp_threshold": 0.75
        }
        
        # 可视化颜色配置
        self.class_colors = ['#FFB6C1', '#9370DB', '#90EE90']  # 浅粉, 紫色, 浅绿
        self.cmap = ListedColormap(self.class_colors)
        
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
        """数据增强：添加高斯噪声和时间扭曲"""
        X_aug = [X]
        y_aug = [y]
        physics_aug = [physics_features]
        
        for _ in range(num_augment):
            # 添加高斯噪声
            X_noise = X + np.random.normal(0, 0.01, X.shape)
            X_aug.append(X_noise)
            y_aug.append(y)
            physics_aug.append(physics_features)
            
            # 时间扭曲
            for i in range(X.shape[0]):
                # 随机选择时间步进行拉伸和压缩
                scale = 1 + np.random.uniform(-0.1, 0.1)
                new_length = int(self.config.timesteps * scale)
                if new_length <= 0:
                    continue
                    
                # 重采样
                X_temp = np.zeros((self.config.timesteps, self.config.params))
                for j in range(self.config.params):
                    orig_signal = X[i, :, j]
                    new_signal = np.interp(
                        np.linspace(0, len(orig_signal)-1, new_length),
                        np.arange(len(orig_signal)),
                        orig_signal
                    )
                    # 截断或填充
                    if new_length > self.config.timesteps:
                        X_temp[:, j] = new_signal[:self.config.timesteps]
                    else:
                        X_temp[:new_length, j] = new_signal
                        X_temp[new_length:, j] = new_signal[-1]  # 填充最后一个值
                X_aug.append(X_temp[np.newaxis, ...])
                y_aug.append(y[i])
                physics_aug.append(physics_features[i])
        
        return np.vstack(X_aug), np.hstack(y_aug), np.vstack(physics_aug)

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

# ======================== 物理引导的损失函数 ========================
class PhysicsGuidedLoss(nn.Module):
    def __init__(self, config, alpha=0.3):
        super().__init__()
        self.config = config
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, logits, targets, sensor_data):
        # 标准交叉熵损失
        ce_loss = self.ce_loss(logits, targets)
        
        # 物理约束损失
        phys_loss = self.compute_physics_loss(sensor_data)
        
        return ce_loss + self.alpha * phys_loss
    
    def compute_physics_loss(self, sensor_data):
        """计算物理约束损失"""
        # 输入形状: (batch, timesteps, params)
        batch_size, timesteps, _ = sensor_data.shape
        
        # 1. 压力-流量关系 (dp/dt ∝ dQ/dt)
        pressure = sensor_data[:, :, self.config.param_groups["pressure"]].mean(dim=2)
        flow = sensor_data[:, :, self.config.param_groups["flow"]].mean(dim=2)
        
        # 计算导数 (使用中心差分)
        dp_dt = pressure[:, 2:] - pressure[:, :-2]  # 中心差分
        dq_dt = flow[:, 2:] - flow[:, :-2]
        
        # 物理关系约束
        phys_loss1 = F.mse_loss(
            dp_dt, 
            self.config.phys_params["pressure_flow_ratio"] * dq_dt
        )
        
        # 2. 温度-功率平衡
        temperature = sensor_data[:, :, self.config.param_groups["temperature"]].mean(dim=2)
        power = sensor_data[:, :, self.config.param_groups["other"]].mean(dim=2)
        
        # 计算导数
        dT_dt = temperature[:, 1:] - temperature[:, :-1]
        dP_dt = power[:, 1:] - power[:, :-1]
        
        # 物理关系约束
        phys_loss2 = F.mse_loss(
            dT_dt, 
            self.config.phys_params["temp_power_ratio"] * dP_dt
        )
        
        return phys_loss1 + phys_loss2

# ======================== 模型训练器 ========================
class NuclearModelTrainer:
    def __init__(self, model, config, train_loader, val_loader):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = config.device
        self.history = {
            "train_loss": [], "val_loss": [],
            "train_acc": [], "val_acc": []
        }
        self.best_val_loss = float('inf')
        self.epochs_no_improve = 0
        
        # 损失函数和优化器
        self.criterion = PhysicsGuidedLoss(config)
        self.optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=config.learning_rate,
            weight_decay=1e-4
        )
        # 移除了verbose参数
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        
        # 将模型移到设备
        self.model.to(self.device)
    
    def train(self):
        """训练模型"""
        print(f"\nStarting training for {self.config.epochs} epochs...")
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
                loss = self.criterion(logits, labels, x)
                
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
                torch.save(self.model.state_dict(), self.config.best_model_save_path)
                print(f"Saved best model with val loss: {val_loss:.4f}")
            else:
                self.epochs_no_improve += 1
                if self.epochs_no_improve >= self.config.patience:
                    print(f"Early stopping after {self.config.patience} epochs without improvement.")
                    break
        
        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time/60:.2f} minutes")
        
        # 加载最佳模型
        self.model.load_state_dict(torch.load(self.config.best_model_save_path))
        print(f"Loaded best model with validation loss: {self.best_val_loss:.4f}")
    
    def evaluate(self):
        """评估模型性能"""
        self.model.eval()
        total_loss, correct, total = 0, 0, 0
        
        with torch.no_grad():
            for x, physics, labels in self.val_loader:
                x, physics, labels = x.to(self.device), physics.to(self.device), labels.to(self.device)
                
                logits, _, _ = self.model(x, physics)  # 忽略特征和解释
                loss = self.criterion.ce_loss(logits, labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(logits, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100 * correct / total
        return avg_loss, accuracy
    
    def test(self, test_loader):
        """在测试集上评估模型"""
        self.model.eval()
        all_preds = []
        all_labels = []
        all_explanations = []
        all_features = []
        
        with torch.no_grad():
            for x, physics, labels in test_loader:
                x, physics, labels = x.to(self.device), physics.to(self.device), labels.to(self.device)
                
                logits, explanations, features = self.model(x, physics)
                _, preds = torch.max(logits, 1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_explanations.extend(explanations)
                all_features.append(features.cpu())
        
        # 计算准确率
        accuracy = 100 * np.mean(np.array(all_preds) == np.array(all_labels))
        print(f"\nTest Accuracy: {accuracy:.2f}%")
        
        # 合并特征
        all_features = torch.cat(all_features, dim=0).numpy()
        
        return all_preds, all_labels, all_explanations, all_features
    
    def visualize_diagnostics(self, test_loader, sample_idx=0):
        """可视化诊断结果"""
        self.model.eval()
        with torch.no_grad():
            for i, (x, physics, labels) in enumerate(test_loader):
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
            f"Predicted: {self.config.fault_types[pred_label]}",
            fontsize=16, fontweight='bold', y=0.98
        )
        
        # 压力参数
        plt.subplot(4, 1, 1)
        plt.plot(sensor_data[:, 45], 'r-', linewidth=1.5)
        plt.title("Primary Pressure", fontsize=12, fontweight='bold')
        plt.ylabel("Normalized Value")
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # 流量参数
        plt.subplot(4, 1, 2)
        plt.plot(sensor_data[:, 5], 'b-', linewidth=1.5)
        plt.title("Coolant Flow Rate", fontsize=12, fontweight='bold')
        plt.ylabel("Normalized Value")
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # 温度参数
        plt.subplot(4, 1, 3)
        plt.plot(sensor_data[:, 25], 'g-', linewidth=1.5)
        plt.title("Core Temperature", fontsize=12, fontweight='bold')
        plt.ylabel("Normalized Value")
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # 解释文本
        plt.subplot(4, 1, 4)
        plt.axis('off')
        plt.text(0.05, 0.5, explanation, fontsize=12, 
                bbox=dict(facecolor='lightyellow', alpha=0.5, boxstyle='round,pad=0.5'))
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig('nuclear_diagnosis.png', dpi=600, bbox_inches='tight')
        plt.show()
    
    def plot_training_history(self):
        """绘制训练历史"""
        plt.figure(figsize=(12, 10))
        
        # 损失曲线
        plt.subplot(2, 1, 1)
        plt.plot(self.history["train_loss"], 'b-', label='Training Loss', linewidth=2)
        plt.plot(self.history["val_loss"], 'r-', label='Validation Loss', linewidth=2)
        plt.title('Training and Validation Loss', fontsize=14, fontweight='bold')
        plt.xlabel('Epochs', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.legend(fontsize=11)
        plt.grid(True, linestyle='--', alpha=0.6)
        
        # 准确率曲线
        plt.subplot(2, 1, 2)
        plt.plot(self.history["train_acc"], 'b-', label='Training Accuracy', linewidth=2)
        plt.plot(self.history["val_acc"], 'r-', label='Validation Accuracy', linewidth=2)
        plt.title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
        plt.xlabel('Epochs', fontsize=12)
        plt.ylabel('Accuracy (%)', fontsize=12)
        plt.legend(fontsize=11)
        plt.grid(True, linestyle='--', alpha=0.6)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=600, bbox_inches='tight')
        plt.show()
    
    def plot_tsne_3d(self, features, labels):
        """绘制3D t-SNE可视化图"""
        print("\nGenerating 3D t-SNE visualization...")
        
        # 使用t-SNE降维到3D
        tsne = TSNE(n_components=3, perplexity=min(30, len(features)-1), 
                   random_state=42, n_iter=3000)
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
                s=70,
                alpha=0.8,
                depthshade=True
            )
        
        # 设置标签和标题
        ax.set_xlabel('t-SNE Dimension 1', fontsize=12, labelpad=10)
        ax.set_ylabel('t-SNE Dimension 2', fontsize=12, labelpad=10)
        ax.set_zlabel('t-SNE Dimension 3', fontsize=12, labelpad=10)
        ax.set_title('3D t-SNE Visualization of Fault Diagnosis Features', 
                    fontsize=16, fontweight='bold', pad=20)
        
        # 添加图例
        ax.legend(fontsize=12, loc='upper right')
        
        # 美化布局
        plt.tight_layout()
        
        # 保存高分辨率图片
        plt.savefig(self.config.tsne_plot_path, dpi=600, bbox_inches='tight')
        print(f"3D t-SNE visualization saved to {self.config.tsne_plot_path}")
        plt.show()
    
    def plot_3d_confusion_matrix(self, y_true, y_pred):
        """绘制3D混淆矩阵"""
        print("\nGenerating 3D confusion matrix...")
        
        # 计算混淆矩阵
        cm = confusion_matrix(y_true, y_pred)
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
        zpos = np.zeros(len(xpos) * len(y_labels))
        
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
        ax.set_title('3D Confusion Matrix of Fault Diagnosis', 
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
        plt.savefig(self.config.confusion_matrix_3d_path, dpi=600, bbox_inches='tight')
        print(f"3D confusion matrix saved to {self.config.confusion_matrix_3d_path}")
        plt.show()

# ======================== 主函数 ========================
def main():
    # 初始化配置
    config = Config()
    print(config)
    print("=" * 80)
    
    start_time = time.time()
    
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
        print("Augmenting data...")
        X_aug, y_aug, physics_aug = signal_processor.augment_data(X_smooth, y, physics_features, num_augment=5)
        print(f"Data augmented from {X_smooth.shape[0]} to {X_aug.shape[0]} samples")
        
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
        
        # 3. 初始化模型
        print("\n[3/5] INITIALIZING MODEL")
        model = PhysicsGuidedNuclearModel(config)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model parameters: {total_params:,} (Trainable: {trainable_params:,})")
        
        # 4. 训练模型
        print("\n[4/5] TRAINING MODEL")
        trainer = NuclearModelTrainer(model, config, train_loader, val_loader)
        trainer.train()
        
        # 5. 评估与可视化
        print("\n[5/5] EVALUATING AND VISUALIZING RESULTS")
        # 保存模型
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': vars(config),
            'history': trainer.history
        }, config.model_save_path)
        print(f"Model saved to {config.model_save_path}")
        
        # 测试模型
        preds, labels, explanations, features = trainer.test(test_loader)
        
        # 可视化训练历史
        trainer.plot_training_history()
        
        # 可视化诊断结果
        trainer.visualize_diagnostics(test_loader)
        
        # 绘制3D t-SNE可视化
        trainer.plot_tsne_3d(features, np.array(labels))
        
        # 绘制3D混淆矩阵
        trainer.plot_3d_confusion_matrix(np.array(labels), np.array(preds))
        
        # 保存部分解释结果
        sample_explanations = explanations[:min(5, len(explanations))]
        with open(config.sample_explanations_path, "w") as f:
            for i, exp in enumerate(sample_explanations):
                f.write(f"Sample {i+1}:\n{exp}\n{'='*80}\n\n")
        print(f"Sample explanations saved to {config.sample_explanations_path}")
        
        # 计算执行时间
        total_time = time.time() - start_time
        print(f"\nDiagnosis completed successfully in {total_time/60:.2f} minutes!")
        
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