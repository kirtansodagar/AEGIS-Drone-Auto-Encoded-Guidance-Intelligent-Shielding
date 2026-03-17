# 🛡️ AEGIS-Drone: Auto-Encoded Guidance & Intelligent Shielding

### *High-Efficiency Vision-Based Autonomous UAV Navigation using Hybrid AI*

> **Published Research** — Department of Computer Science & Engineering, SVNIT Surat
> 📄 Paper available on request: kirtan.sodagar25@gmail.com

**AEGIS-Drone** is a hybrid autonomous navigation framework that enables quadcopters to fly intelligently inside complex environments using only depth camera input. Built on top of **Microsoft AirSim**, AEGIS solves the major bottleneck of raw image processing by decoupling perception from control through a highly efficient autoencoder.

---

## 🏆 Key Results

| Metric | AEGIS-Drone | CNN Baseline |
|--------|:-----------:|:------------:|
| Navigation Success Rate | **85%** | 42% |
| Collisions (50 episodes) | **9** | 31 |
| Perception Latency | **8–12 ms** | 120 ms |
| Control Loop Frequency | **25–30 Hz** | ~8 Hz |
| Path Efficiency (η) | **0.68** | 0.41 |
| RL Convergence | **50k–70k steps** | 200k+ steps |

> AEGIS achieves **2× higher success rate**, **60% fewer collisions**, and **10–15× lower latency** vs end-to-end CNN-PPO baseline across 50 fully randomised AirSim Blocks episodes.

![Training Reward Curve](results/reward_plot.png)

---

## 🚀 Project Overview

AEGIS uses a **four-layer Hybrid AI Architecture** — each layer solving a distinct failure mode of end-to-end RL:

| Layer | Component | Role |
|-------|-----------|------|
| 👁️ **Perception** | Convolutional Autoencoder | Compresses 64×64 depth → 128-dim latent vector |
| 🧠 **Decision** | PPO (MLP Policy) | Learns optimal navigation in latent space |
| ⚖️ **Stability** | PID Controller | Maintains steady altitude at –10 m |
| 🛡️ **Safety** | Quadrant Safety Shield | Detects obstacles < 4 m, triggers evasive maneuvers |

### Why this works better than end-to-end RL

End-to-end CNN-PPO must solve perception, control, stability, and safety simultaneously — leading to slow convergence, high latency, and unsafe exploration. AEGIS separates each concern:

- **Autoencoder** removes pixel noise → 10–15× faster inference
- **Latent-space PPO** trains on smooth 128-dim vectors → 3–5× faster convergence
- **PID** handles altitude → PPO only learns lateral navigation
- **Safety Shield** prevents catastrophic crashes during training → longer episodes, richer gradients

---

## 🏗️ System Architecture

```
Depth Camera (64×64)
        │
        ▼
┌─────────────────┐
│  Autoencoder    │  → 128-dim latent vector (2–4 ms)
└─────────────────┘
        │
        ▼
┌─────────────────┐     ┌─────────────────┐
│   PPO Policy    │     │  Safety Shield  │  ← raw depth (quadrant analysis)
│  (MLP, latent)  │     │  override if    │
│  → vx, vy, yaw  │     │  obstacle < 4m  │
└─────────────────┘     └─────────────────┘
        │                       │
        └──────────┬────────────┘
                   ▼
        ┌─────────────────┐
        │ Action Arbiter  │  Safety > PPO > PID priority
        └─────────────────┘
                   │
                   ▼
        ┌─────────────────┐
        │  PID Altitude   │  → thrust command
        └─────────────────┘
                   │
                   ▼
            Final UAV Command
           (roll/pitch/yaw/throttle)
```

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

1. Download **Blocks.zip** from the [AirSim Releases Page](https://github.com/microsoft/AirSim/releases).
2. Extract and run `Blocks.exe`.
3. Choose **"No"** when asked about car simulation to enable quadrotor mode.

### 2. Clone the Repository (WSL2)

```bash
git clone https://github.com/kirtansodagar/AEGIS-Drone-Auto-Encoded-Guidance-Intelligent-Shielding.git
cd AEGIS-Drone-Auto-Encoded-Guidance-Intelligent-Shielding
```

### 3. Create Environment & Install Dependencies

```bash
python3 -m venv aegis_env
source aegis_env/bin/activate
pip install -r requirements.txt
```

---

## 🏃 Usage Guide

### ▶ Option A: Autonomous Flight Demo (Recommended)

```bash
python3 src/training/evaluate_ultimate_autonomous_v3.py
```

The drone will take off, hold altitude via PID, navigate forward, and avoid obstacles using the safety shield.

### ▶ Option B: Full Training Pipeline

```bash
# Phase 1 — Collect depth images
python3 src/training/collect_data.py

# Phase 2 — Train autoencoder
python3 src/training/train_autoencoder.py
# Output: autoencoder_encoder.pth

# Phase 3 — Train PPO policy on frozen encoder
python3 src/training/airsim_mlp_train.py
# Output: saved models in airsim_mlp_logs/
```

---

## 📂 Project Structure

```
AEGIS-Drone/
├── src/
│   └── training/
│       ├── airsim_mlp_env.py           # Custom AirSim Gym environment
│       ├── collect_data.py             # Depth image data collection
│       ├── train_autoencoder.py        # Autoencoder training
│       ├── airsim_mlp_train.py         # PPO policy training
│       ├── evaluate_ultimate_autonomous_v3.py  # Full demo
│       └── autoencoder_encoder.pth     # Trained encoder weights
├── airsim_mlp_logs/
│   └── airsim_ppo_mlp_model_50000_steps.zip   # Trained PPO model
├── results/
│   └── reward_plot.png                # Training reward curve
├── requirements.txt
└── README.md
```

---

## 🔬 Ablation Study

All 4 variants tested across 50 randomised episodes:

| Variant | Success % | Collisions | Latency | Path η |
|---------|:---------:|:----------:|:-------:|:------:|
| End-to-End CNN | 42% | 31 | 120 ms | 0.41 |
| AEGIS w/o Autoencoder | 55% | 22 | 75 ms | — |
| AEGIS w/o PID | 61% | 19 | 8–12 ms | 0.49 |
| AEGIS w/o Shield | 68% | 24 | 8–12 ms | 0.52 |
| **Full AEGIS (ours)** | **85%** | **9** | **8–12 ms** | **0.68** |

---

## 👥 Contributors

- **Sodagar Kirtan Rajesh** — AI Architecture, RL Implementation, Autoencoder Training, Simulation Setup, WSL–Windows Integration
- **Maythreyi Vijaykumar** — Research Co-author, Department of CSE, SVNIT Surat

---

## 📄 License

This repository is licensed under the **MIT License**.

---

## 📬 Contact

**Kirtan Sodagar** — kirtan.sodagar25@gmail.com | [GitHub](https://github.com/kirtansodagar)

*For the full research paper, system weights, or collaboration inquiries, feel free to reach out.*
