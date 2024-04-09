# Post CVPR Update:

0) Cleaning:
   1) File by file. 
   2) Abstraction/refactor?
1) Change Objective to R - alpha |L|. 
2) Add 3D CSG dataset.
3) SA PLAD version.
4) Get RL Working 
5) Speed up execution Batching + Compile
6) Method Update:
   1) Random pretrianing vs rewrite pretraining.
   2) rewrite scheduling
   3) WS or Not? 


# 2D CSG Environment

Goal: Create a Gym environment for training RL agents on CSG data.

* [2021-10-06] Figure out how to do batch simulation - Run multiple environments - handled by SB3 itself.

* [2021-10-08] Refine Env code and define multiple agent models. 


## 2021/12/8 Update:
  
* Updated Results

* Verified cluster usage. 


## Update: 2021/12/7


* Updated Config:
  * Config split into `config.py` and `subconfig.py`.
  * All default values now in `config/basic.py`.
  
* Connected to Cluster.

## Update: 2022/01/13

* Target: Get DQN and PPO with HER/C-HER.
  * Refactor Code
    * Is image utils required?
    * Scheduler
  * Find Minimal PPO which works.
  * Find minimal Off-policy which works.  
* Add data-loader visualizations. to t-board.
* Improve access to training results running on cluster.
* Manage Notebook update. 
* Hypothesis: With CHER we can skip pretraining completely.