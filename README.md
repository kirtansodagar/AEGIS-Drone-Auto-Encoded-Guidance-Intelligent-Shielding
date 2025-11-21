# 🛡️ AEGIS-Drone: Auto-Encoded Guidance & Intelligent Shielding  
### *High-Efficiency Vision-Based Autonomous UAV Navigation using Hybrid AI*

**AEGIS-Drone** is a hybrid autonomous navigation framework that enables quadcopters to fly intelligently inside complex environments using only depth camera input. Built on top of **Microsoft AirSim**, AEGIS solves the major bottleneck of raw image processing by decoupling perception from control through a highly efficient autoencoder.  

This results in training speeds **> 30 FPS** (vs. < 5 FPS for end-to-end CNN training), while preserving stability, safety, and obstacle-avoidance reliability.

---

## 🚀 Project Overview

AEGIS uses a **three-layer Hybrid AI Architecture**:

### 1️⃣ Perception — *“The Eyes”*  
A **Deep Convolutional Autoencoder** compresses `64×64` depth images into a compact **128-dimensional latent vector**, enabling fast policy training.

### 2️⃣ Decision — *“The Pilot”*  
A **PPO (Proximal Policy Optimization)** agent with an **MLP policy** processes latent vectors to navigate efficiently.

### 3️⃣ Safety & Stability — *“The Reflexes”*  
- **PID Controller** stabilizes altitude at **−10 m**  
- **Safety Shield Module** performs deterministic **quadrant analysis** and overrides the agent when obstacles < 4 m are detected  
- Implements automatic *backup* and *strafe* evasive maneuvers  

---

## 📋 Prerequisites

### ✔ System Requirements  
- **OS**: Windows 10/11 (Host) + WSL2 (Ubuntu 20.04/22.04)  
- **GPU**: NVIDIA GTX 1050 or better recommended  
- **Simulator**: Microsoft **AirSim** (pre-compiled binaries)  
- **Python**: `3.8+`  

---

## 🛠 Installation & Setup

### 1. Install AirSim (Windows Host)

1. Download **Blocks.zip** or **AirSimNH.zip** from the AirSim Releases Page.  
2. Extract the folder anywhere (Desktop recommended).  
3. Run `Blocks.exe`.  
4. Choose **“No”** when asked about car simulation to enable quadrotor mode.

---

### 2. Clone the Repository (WSL2)

```bash
git clone https://github.com/YOUR_USERNAME/AEGIS-Drone-Auto-Encoded-Guidance-Intelligent-Shielding.git
cd AEGIS-Drone-Auto-Encoded-Guidance-Intelligent-Shielding
```

### 3. Create Environment & Install Dependencies

```bash
python3 -m venv aegis_env
source aegis_env/bin/activate
pip install -r requirements.txt
```

---

## 🏃‍♂️ Usage Guide

### ▶ Option A: Autonomous Flight Demo (Recommended)

Run this script to see the **final intelligent drone**:

```bash
python3 src/training/evaluate_ultimate_autonomous_v3.py
```

**The drone will:**
- Take off  
- Hold altitude using PID  
- Move forward  
- Avoid obstacles using quadrant-based safety shielding  

---

### ▶ Option B: Full Training Pipeline

#### **Phase 1 — Data Collection**
Collect raw depth images:

```bash
python3 src/training/collect_data.py
```

#### **Phase 2 — Autoencoder Training**
Train the visual perception model:

```bash
python3 src/training/train_autoencoder.py
```

**Output:** `autoencoder_encoder.pth`

#### **Phase 3 — PPO Policy Training**
Train navigation policy on frozen encoder:

```bash
python3 src/training/airsim_mlp_train.py
```

**Output:** saved models in `airsim_mlp_logs/`

---

## 📂 Project Structure

```text
AEGIS-Drone-Auto-Encoded-Guidance-Intelligent-Shielding/
│
├── src/
│   ├── training/
│   │   ├── airsim_mlp_env.py
│   │   ├── collect_data.py
│   │   ├── train_autoencoder.py
│   │   ├── airsim_mlp_train.py
│   │   ├── evaluate_ultimate_autonomous_v3.py
│   │   └── autoencoder_encoder.pth
│
├── airsim_mlp_logs/
│   └── airsim_ppo_mlp_model_50000_steps.zip
│
├── results/
│   └── reward_plot.png
│
├── requirements.txt
└── README.md
```

---

## 🔬 Core Algorithms

| Layer | Algorithm | Description |
|-------|-----------|-------------|
| **Decision** | PPO (MLP Policy) | Learns optimal navigation actions |
| **Perception** | Convolutional Autoencoder | Converts `64×64` image → 128-dim vector |
| **Stability** | PID Controller | Maintains steady altitude |
| **Safety** | Quadrant Depth Analysis | Detects obstacles & computes safe escape direction |

---

## 👥 Contributors

- **SODAGAR KIRTAN RAJESH** — AI Architecture, RL Implementation, Autoencoder Training  
Simulation Setup, Networking, WSL–Windows Integration  

---

## 📝 License

This repository is licensed under the **MIT License**.

