# Project 2: Continuous Control

### Introduction


This repo is for solving the "Reacher" unity game engine gym. One environment contains a robot arm with two joints and a target sphere in which the end effector is to move into. Per step the end effector of the arm is inside the goal sphere, a small reward is given. The gym env variant that this code is solving contains 20 arms which run in lockstep and all have different goals.


### Project Details

Two variants: single arm and twenty arms gyms. We will only consider the 20 arm gym in this project repo.

#### State and action space

The output of the gym env and the input to the policy is a (20, 33) float64 tensor.

It contains:

| dims  |  n  | purpose                             |
| ----- | --- | ----------------------------------- |
| 0:3   | 3   | pendulum A position                 |
| 3:7   | 4   | A rotation, quaternion (x, y, z, w) |
| 7:10  | 3   | A angular velocity                  |
| 10:13 | 3   | A linear velocity                   |
| 13:16 | 3   | pendulum B position                 |
| 16:20 | 4   | B rotation, quaternion (x, y, z, w) |
| 20:23 | 3   | B angular velocity                  |
| 23:26 | 3   | B linear velocity                   |
| 26:29 | 3   | goal position                       |
| 29:32 | 3   | hand local position                 |
| 32    | 1   | goal speed                          |

The output of the policy (action space) and the input of the env is (20, 4) float64 tensor which represent the torques for the 2 arm joints per arm.

| dim | purpose                      |
| --- | ---------------------------- |
| 0   | torque on pendulum A, x-axis |
| 1   | torque on pendulum A, z-axis |
| 2   | torque on pendulum B, x-axis |
| 3   | torque on pendulum B, z-axis |


### Solve condition

From the documentation:

The barrier for solving multi-arm environment is an average score of +30 (over 100 consecutive episodes, and over all agents). Specifically,
- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent.  This yields 20 (potentially different) scores.  We then take the average of these 20 scores. 
- This yields an **average score** for each episode (where the average is over all 20 agents).

The environment is considered solved, when the average (over 100 episodes) of those average scores is at least +30. 


### Installing dependencies

Really only pytorch, numpy and the reacher environment

Create and activate the Conda environment from the repository root:

```bash
conda env create -f environment.yml
conda activate p2-continuous-control
```

That covers the four dependencies the two scripts need: PyTorch, NumPy, Matplotlib
(for the learning curve plot) and the Reacher environment itself. The Reacher
environment is driven through the `unityagents` package (ml-agents 0.4), which is
vendored in the `python/` folder of the Udacity course repo rather than installed
from PyPI, so `environment.yml` only installs the three packages `unityagents` needs
at runtime (`grpcio`, `pillow`, `protobuf==3.20.3`). Two pins matter:
`protobuf==3.20.3`, because the generated `*_pb2.py` files are protoc 3.5.2 era and
protobuf >= 4.21 rejects them, and `numpy<2`, because `unityagents` still uses the
removed `np.float_`.

The Unity binary for the 20-arm variant is included in this repo under
`Reacher_Linux_20/`, which is where both scripts look by default. If it is instead
unpacked next to the repository folder, as in the course repo layout, that location is
used as a fallback. Either way `--binary <path>` overrides it.

The `unityagents` package is not included. It is imported from `../../python` when it
is not installed, which is where it sits when this repo is cloned into the
`p2_continuous-control` folder of the Udacity course repo.




