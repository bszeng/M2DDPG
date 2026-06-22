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
> **Quick Start:** Download the dataset from the Baidu link above, then proceed to **Step 1** to visualize the results immediately.

**0. Requirements**
  ```bash
  matplotlib==3.7.2
  numpy==1.24.3
  torch==1.10.0
  ```

**1. Plot curves**

To generate the figures, open and run `plot_curves.ipynb` located in that directory. The required result files (`.npy`) are saved in the `Learning Curves` folder. 

**2. Train the Model from Scratch (Reproduce: `Learning Curves/`)**

Train each algorithm to generate the learning curves.
Note that the MI ablation and M²DDPG require the meta-initialization parameters (`Models/`) from **Step 4**, which in turn depends on the multi-task buffer from **Step 3**.
```bash
# Run vanilla DDPG benchmark
python main_Benchmark.py --num_RIS_elements 4 --lr 1e-2

# Run Meta-Critic (MC) ablation
python main_MC.py --num_RIS_elements 4 --lr 1e-2

# Run Meta-Initialization (MI) ablation  (requires Step 4)
python main_Meta.py --mode "validation" --num_RIS_elements 4 --lr 1e-2

# Run full M²DDPG (Double Meta) (requires Step 4)
python main_MMC.py --num_RIS_elements 4 --lr 1e-2
```

**3. Construct Pre-Collection Multi-Task Dataset (Reproduce: `MultiTask Buffer/`)**

Construct the pre-collection multi-task dataset via `main_Buffer.py`. 
This script collects experience tuples from multiple RIS-aided MU-MISO tasks with varying channel conditions and stores them in `MultiTask Buffer/`.
```bash
python main_Buffer.py --num_RIS_elements 4 --num_eps 10000 --num_time_steps_per_eps 1000 --buffer_size 500
```

**4. Train Meta-Initialization Parameters (Reproduce: `Models/`, based on `MultiTask Buffer/`)**

Using the pre-collected multi-task dataset from **Step 3**, train the meta-initialization parameters via the Reptile algorithm. 
The resulting transferable actor and critic network parameters are saved in `Models/`.
```bash
python main_Meta.py --mode "train" --num_RIS_elements 4 --lr 1e-3
```
