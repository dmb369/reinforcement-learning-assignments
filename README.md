# Reinforcement Learning — Programming Assignments

Coursework for a Reinforcement Learning course, PA1 through PA5. Each folder
is a self-contained assignment with its own code, saved outputs, and
(where provided) the report submitted for grading.

## Repo structure

```
RL/
├── Textbook.pdf
├── PA1/                       Multi-armed / contextual bandits
│   ├── lin_ucb.py
│   ├── lin_greedy.py
│   ├── policy_gradient.py
│   ├── RL graphs/             reward curves, per-algorithm plots
│   └── Report.pdf
├── PA2/                       MDP value/policy iteration
│   ├── Value_Iteration.py
│   ├── policy_iteration.py
│   ├── policy_evaluation.py
│   ├── Assignment2Tools.py    shared helpers (Markov matrix / prob-vector generation)
│   └── Report.pdf
├── PA3/                       Custom Gym environment
│   ├── GymTraffic.py          custom traffic-light control environment
│   ├── training.py
│   ├── policy1.npy / policy2.npy / policy3.npy
│   └── programming_assignment_3.pdf
├── PA4/                       Offline vs. online RL
│   ├── online_data.py
│   ├── DQN_offline_false.keras
│   ├── car_dataset.csv
│   └── *.png                  training curves
└── PA5/                       Team project (Stable-Baselines3)
    ├── training1.py           DQN on LunarLander-v2
    ├── training2.py           reward-shaped DQN on MountainCar-v0
    ├── training3.py           DQN on a custom RobotNavigationEnv
    ├── training4.py           hierarchical TD3 on the same env
    └── contribution.txt       team + per-section contributions
```

## PA1 — Contextual bandits for cloud job allocation

Three bandit algorithms — **LinUCB**, **linear ε-greedy**, and a
**policy-gradient** agent — are trained on a custom `ServerAllocationEnv`
(cloud job-scheduling environment), using hand-engineered features (job
count, average priority, job-type counts, average network load) extracted
from the raw observation. Results and reward curves for each algorithm are
in `RL graphs/` and `Report.pdf`.

## PA2 — MDP value/policy iteration

Implements tabular **value iteration**, **policy iteration**, and **policy
evaluation** for a resource-allocation MDP with state variables `(x, z, b,
t)` and action-dependent transition/reward structure. `Assignment2Tools.py`
generates the probability vectors and Markov transition matrices the
solvers rely on. See `Report.pdf` for the full state/action space
definition and results.

## PA3 — Custom traffic-light control environment

`GymTraffic.py` implements a custom `gym.Env` (`GymTrafficEnv`) modeling a
two-road traffic intersection with stochastic arrivals, a switchable
signal, and a minimum-hold constraint on switching. `training.py` trains a
policy against it; `policy1/2/3.npy` are saved trained policies for
different settings. See `programming_assignment_3.pdf` for the problem
spec.

## PA4 — Offline vs. online DQN

Compares an offline-trained DQN against online data collection on a
car-control task (`car_dataset.csv`). `online_data.py` loads/relabels the
offline transitions and trains a Keras DQN (`DQN_offline_false.keras`);
plots comparing online vs. offline performance are included.

## PA5 — Team project: Stable-Baselines3 (DQN, reward shaping, hierarchical TD3)

A 4-person team project reusing a shared `training*.py` scaffold:
- `training1.py` — DQN (Stable-Baselines3) on **LunarLander-v2**, replacing
  a starter PPO agent, with a custom two-layer policy network.
- `training2.py` — reward shaping on **MountainCar-v0** via a custom
  Gymnasium wrapper (shaped reward = weighted position/velocity/energy),
  still trained with DQN, evaluated against the original sparse reward.
- `training3.py` — DQN on a custom **RobotNavigationEnv**, trained for 2M
  timesteps with a custom logging/eval/checkpoint callback.
- `training4.py` — a **hierarchical** variant of the same navigation task:
  converts discrete control into continuous intermediate-goal planning and
  trains with **TD3**, with reward shaping based on Euclidean distance to
  sub-goals and collision penalties.

Full per-member contribution breakdown is in `contribution.txt`.

## Setup

```bash
pip install numpy scipy matplotlib gymnasium gym torch tensorflow stable-baselines3 pygame
```
(Exact versions weren't pinned in the original coursework; install
whatever versions are current for the imports used in each script.)

## Running an assignment

Each script is standalone — from the relevant `PA*/` folder:
```bash
python3 lin_ucb.py            # e.g. PA1
python3 Value_Iteration.py    # e.g. PA2
python3 training1.py          # e.g. PA5
```
Note: PA1's scripts import `ServerAllocationEnv` from a `CloudComputing`
module, and PA3 imports `gym`/`gymnasium` environment definitions that were
provided as course starter code — if these aren't included in your copy of
the assignment, you'll need the original course-provided environment file
alongside the scripts.
