PSI-MDL: Physical Soft-constraint Interpretable Multimodal Deep Learning for Nuclear Power Plant Fault Diagnosis
https://img.shields.io/badge/license-MIT-blue.svg
https://img.shields.io/badge/DOI-10.xxxx/xxxxx-brightgreen

Official implementation of the paper:
A Physical Soft-constraint Interpretability-based Multimodal Deep Learning Model for Small-sample Fault Diagnosis of Pressurized Water Reactor Coolant Systems in Nuclear Power Plants

📖 Introduction
This repository contains the code for a novel multimodal deep learning framework designed for fault diagnosis in nuclear power plant coolant systems. The model addresses two critical challenges in nuclear safety:

Extreme scarcity of fault samples (only 30 original samples available)

Black-box nature of traditional deep learning models

Key innovations include:

🧪 Physics-guided data augmentation: Expands 30 samples to 330 physically consistent samples using pressure-flow constraints, non-negativity constraints, and temperature boundaries

🧠 Multimodal fusion architecture: Integrates Mamba blocks for temporal modeling, CNN for spatial feature extraction, and physics processors for domain knowledge

⚖️ Differentiable physical constraints: Embeds energy/mass conservation laws as soft regularization terms in loss function

💬 Natural language explanations: Generates human-readable diagnostic reports with 96.7% expert agreement
✨ Key Features
11x sample expansion while maintaining physical consistency

99.97% diagnostic accuracy with only 0.192M parameters

32% reduction in physical constraint violations

Triple interpretability:

Feature importance weights

Physical rule verification

Natural language explanations

Dynamic hybrid loss optimization: Balances data-driven and physics-based learning
📂 Dataset
The dataset is generated using the GSE GPWR Generalized Nuclear Simulator (GNS) certified to ANS 3.5 standards. It contains:

3 fault types: Cold Leg LOCA, Hot Leg LOCA, Small Break

61 time steps (0-60 seconds)

88 sensor parameters per time step

5 engineered physical features:

Pressure-Flow Ratio (PFR)

Max Temperature Gradient (MTG)

Flow Imbalance (FI)

Pressure Variation (VT)

Temperature-Power Ratio (TPR)

To request dataset access, please contact the corresponding author.
🚀 Usage
Training
bash
python proposed1.py
