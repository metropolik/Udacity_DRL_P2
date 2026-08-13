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

Create and activate the conda env:

```bash
conda env create -f environment.yml
conda activate p2-continuous-control
pip install --no-deps unityagents==0.4.0
```

Matplotlib is in there as well, for the learning curve plot. The rest (grpcio, pillow,
protobuf) is only what `unityagents` needs to talk to the Unity binary. Two versions
are pinned: protobuf 3.20.3, because the generated `*_pb2.py` files are too old for
protobuf 4, and numpy < 2, because `unityagents` still uses `np.float_`.

The Reacher binary for the 20 arm variant is in `Reacher_Linux_20/`. Both scripts look
there first and fall back to `../Reacher_Linux_20`. `--binary` overrides it.





