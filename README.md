# M²DDPG
Dataset download link: https://pan.baidu.com/s/1c9U0mtchy7tPOlB_Jt1qhg?pwd=2026

This repository solves the joint beamforming and phase-shift optimization problem in a Reconfigurable Intelligent Surface (RIS) Assisted Multiuser Multi-Input Single-Output (MU-MISO) system (https://github.com/baturaysaglam/RIS-MISO-Deep-Reinforcement-Learning) using the **Double Meta-Deep Deterministic Policy Gradient (M²DDPG)** framework.

M²DDPG synergizes two complementary meta-learning mechanisms: (i) **offline meta-initialization** via the Reptile algorithm to learn a transferable parameter initialization across diverse tasks, and (ii) **online meta-critic adaptation** to provide auxiliary gradients for rapid task-specific adaptation via bi-level optimization.

## Repository Structure
```
├── Learning Curves/        # Saved training and evaluation results
├── Models/                 # Meta-initialization (Reptile) 
├── MultiTask Buffer/       # Pre-collected multi-task training datasets
├── DDPG.py                 # Baseline DDPG agent
├── DDPG_Buffer.py          # Experience replay buffer
├── DDPG_MC.py              # Meta-Critic (MC) ablation agent
├── DDPG_Meta.py            # Meta-Initialization (MI) ablation agent
├── DDPG_MMC.py             # M²DDPG: Double Meta agent
├── environment.py          # RIS-aided MU-MISO environment
├── main_Benchmark.py       # Run vanilla DDPG benchmark
├── main_Buffer.py          # Construct buffer pre-collection
├── main_MC.py              # Run MC ablation
├── main_Meta.py            # Run MI ablation
├── main_MMC.py             # Run full M²DDPG
├── utils.py                # Utility functions
└── README.md
```

## Run
**0. Requirements**
  ```bash
  matplotlib==3.7.2
  numpy==1.24.3
  torch==1.10.0
  ```
**1. Train the model from scratch**

* Construct pre-collection Multi-Task Buffer (**or download from the dataset link above**)

Before running the full M²DDPG, pre-collect the multi-task dataset:
```bash
python main_Buffer.py --num_antennas 4 --num_RIS_elements 4 --num_users 4 --num_eps 10000 --num_time_steps_per_eps 1000 --buffer_size 500
```
  * Run benchmark (MC, MI and M²DDPG)
   ```
   python main_Benchmark.py --num_antennas 4 --num_RIS_elements 4 --num_users 4
  ```
**2. Plot curves**
* Upon completion, the result files (`.npy`) are saved in the `Learning Curves` folder (**or download from the dataset link above**). 
To generate the figures, open and run `plot_curves.ipynb` located in that directory.
