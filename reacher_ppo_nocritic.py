import argparse
import math
import os
import sys
import time

import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict

matplotlib.use("Agg")

HERE = os.path.dirname(os.path.abspath(__file__))
P2_DIR = os.path.dirname(HERE)
REPO_ROOT = os.path.dirname(P2_DIR)

try:
    from unityagents import UnityEnvironment
except ImportError:  # the package is vendored, not installed
    sys.path.insert(0, os.path.join(REPO_ROOT, "python"))
    from unityagents import UnityEnvironment

DEFAULT_BINARY = os.path.join(HERE, "Reacher_Linux_20", "Reacher.x86_64")
if not os.path.exists(DEFAULT_BINARY):  # not vendored, sitting next to the repo
    DEFAULT_BINARY = os.path.join(P2_DIR, "Reacher_Linux_20", "Reacher.x86_64")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ReacherEnv:
    """
    One Unity process holds all 20 arms and steps them in lockstep
    Every call returns arrays whose first axis is the agent.
    """

    def __init__(self, file_name=DEFAULT_BINARY, seed=0, no_graphics=True,
                 worker_id=0):
        self.env = UnityEnvironment(
            file_name=file_name, seed=seed, no_graphics=no_graphics,
            worker_id=worker_id,
        )
        self.brain_name = self.env.brain_names[0]
        brain = self.env.brains[self.brain_name]
        info = self.env.reset(train_mode=True)[self.brain_name]

        self.num_envs = len(info.agents)
        self.action_size = brain.vector_action_space_size
        self.state_size = info.vector_observations.shape[1]

    def reset(self, train_mode=True):
        info = self.env.reset(train_mode=train_mode)[self.brain_name]
        return info.vector_observations

    def step(self, actions):
        info = self.env.step(np.clip(actions, -1.0, 1.0))[self.brain_name]
        return (
            info.vector_observations,
            np.asarray(info.rewards, dtype=np.float64),
            np.asarray(info.local_done, dtype=bool),
        )

    def close(self):
        self.env.close()


class Policy(nn.Module):
    def __init__(self, state_size=33, action_size=4, hidden=128, std0=0.5):
        super(Policy, self).__init__()

        self.fc_net = nn.Sequential(
            OrderedDict(
                [
                    ("fc1", nn.Linear(state_size, hidden)),
                    ("relu1", nn.ReLU()),
                    ("fc2", nn.Linear(hidden, hidden)),
                    ("relu2", nn.ReLU()),
                    ("fc3", nn.Linear(hidden, action_size)),
                ]
            )
        )
        self.tanh = nn.Tanh()

        # state-independent spread, one per action dim, learned
        self.log_std = nn.Parameter(
            torch.full((action_size,), math.log(std0), dtype=torch.float32)
        )
        self.log_std_min, self.log_std_max = -2.0, 0.5

        # He init on the first two layers
        self.apply(self._he_init)
        # Uniform init on the last layer with 0 bias
        nn.init.uniform_(self.fc_net.fc3.weight, -0.01, 0.01)
        nn.init.zeros_(self.fc_net.fc3.bias)

    @staticmethod
    def _he_init(module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x):
        """states -> (mean, std)"""
        mean = self.tanh(self.fc_net(x))
        std = torch.exp(
            torch.clamp(self.log_std, self.log_std_min, self.log_std_max)
        )
        return mean, std.expand_as(mean)

    def distribution(self, x):
        mean, std = self(x)
        return torch.distributions.Normal(mean, std)

    def act(self, states):
        """Sample actions for one timestep. Returns detached numpy, as pi_old."""
        with torch.no_grad():
            dist = self.distribution(states)
            actions = dist.sample()
            log_probs = dist.log_prob(actions).sum(-1)
        return actions.cpu().numpy(), log_probs.cpu().numpy()


def states_to_log_prob(policy, states, actions):
    """The states_to_prob of this experiment: (T, n, S) -> (T, n) log pi(a|s)."""
    flat_states = states.view(-1, states.shape[-1])
    flat_actions = actions.view(-1, actions.shape[-1])

    dist = policy.distribution(flat_states)
    log_probs = dist.log_prob(flat_actions).sum(-1)

    return log_probs.view(states.shape[:-1])


def collect_trajectories(envs, policy, tmax=1001):
    """
    Roll out policy to collect new trajectories

    (log_prob_list, state_list, action_list, reward_list); the log-probs are
    detached - they are the pi_old of the surrogate.
    """
    log_prob_list, state_list, action_list, reward_list = [], [], [], []

    states = envs.reset(train_mode=True)

    for t in range(tmax):
        batch_input = torch.as_tensor(
            states, dtype=torch.float32, device=device
        )

        actions, log_probs = policy.act(batch_input)

        next_states, rewards, dones = envs.step(actions)

        state_list.append(states)
        action_list.append(actions)
        log_prob_list.append(log_probs)
        reward_list.append(rewards)

        states = next_states

        # defensive programming
        # to avoid requiring padding, but in this task that should never happen anyway
        if dones.any():
            break

    return log_prob_list, state_list, action_list, reward_list


def clipped_surrogate(policy, old_log_probs, states, actions, rewards,
                      discount=0.99, epsilon=0.1):
    """Returns a SCALAR tensor carrying grad_fn, to be MAXIMIZED.

    Shapes:
        states        (T, n_agents, 33)   raw Unity observations
        actions       (T, n_agents, 4)    unclipped Gaussian samples -> torques for arms
        old_log_probs (T, n_agents)       detached, summed over action dims -> for rescaling
        rewards       (T, n_agents)
    """

    rewards = np.asarray(rewards)                    # (T, n_agents)
    G = rewards.copy()
    # loop t from T-2 to 0 (inclusive)
    for t in reversed(range(len(rewards) - 1)):
        G[t] += discount * G[t + 1]

    G -= G.mean(axis=1, keepdims=True)
    G /= G.std(axis=1, keepdims=True) + 1e-10
    G = torch.from_numpy(G).to(torch.float32).to(device)

    states = torch.as_tensor(np.asarray(states), dtype=torch.float32, device=device)
    actions = torch.as_tensor(np.asarray(actions), dtype=torch.float32, device=device)
    old_log_probs = torch.as_tensor(
        np.asarray(old_log_probs), dtype=torch.float32, device=device
    )

    new_log_probs = states_to_log_prob(policy, states, actions)

    # PPO: ratio = new_probs / old_probs. Same thing, in log space.
    ratio = torch.exp(new_log_probs - old_log_probs)
    l_clip = torch.min(ratio * G, torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * G)

    return torch.mean(l_clip)



def train(
    envs,
    episodes=300,
    tmax=1001,
    discount_rate=0.99,
    epsilon=0.1,
    epsilon_decay=0.999,
    sgd_epoch=4,
    lr=5e-4,
    lr_step=1e-4,
    lr_step_every=500,
    lr_min=1e-4,
    hidden=128,
    std0=0.5,
    seed=1234,
    print_every=10,
    checkpoint_path="reacher_ppo_nocritic.policy",
    plot_path="reacher_ppo_nocritic_scores.png",
    resume_from=None,
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    policy = Policy(
        state_size=envs.state_size,
        action_size=envs.action_size,
        hidden=hidden,
        std0=std0,
    ).to(device)

    if resume_from is not None:
        chkpt = torch.load(resume_from, map_location=device, weights_only=False)
        policy.load_state_dict(chkpt["policy"])
        print("Resuming from checkpoint:", resume_from)

    optimizer = optim.Adam(policy.parameters(), lr=lr)

    print("device:", device)
    print("agents:", envs.num_envs, " tmax:", tmax, " episodes:", episodes,
          " sgd_epoch:", sgd_epoch, " lr:", lr)
    print("lr schedule: -{0:g} every {1:d} episodes, floor {2:g}".format(
        lr_step, lr_step_every, lr_min))

    mean_rewards = []
    t_start = time.time()
    for e in range(episodes):

        current_lr = max(lr - lr_step * (e // lr_step_every), lr_min)
        for group in optimizer.param_groups:
            group["lr"] = current_lr

        # collect trajectories ONCE, then reuse them sgd_epoch times
        old_log_probs, states, actions, rewards = collect_trajectories(
            envs, policy, tmax=tmax
        )

        total_rewards = np.sum(rewards, axis=0)

        for _ in range(sgd_epoch):
            L = -clipped_surrogate(
                policy,
                old_log_probs,
                states,
                actions,
                rewards,
                discount=discount_rate,
                epsilon=epsilon,
            )
            optimizer.zero_grad()
            L.backward()
            optimizer.step()
            del L

        epsilon *= epsilon_decay

        mean_rewards.append(np.mean(total_rewards))

        if (e + 1) % print_every == 0:
            window = mean_rewards[-100:]
            with torch.no_grad():
                std = torch.exp(
                    torch.clamp(policy.log_std, policy.log_std_min,
                                policy.log_std_max)
                ).mean().item()
            print(
                "Episode: {0:d}, score: {1:.3f}, avg100: {2:.3f}, "
                "std: {3:.3f}, lr: {4:g}, {5:.1f}s".format(
                    e + 1, np.mean(total_rewards), np.mean(window), std,
                    current_lr, time.time() - t_start,
                )
            )
            sys.stdout.flush()

        torch.save({"policy": policy.state_dict()}, checkpoint_path)

    print("Saved checkpoint to", checkpoint_path)
    save_score_plot(mean_rewards, plot_path)

    return policy, mean_rewards


def save_score_plot(mean_rewards, output_path):
    import matplotlib.pyplot as plt

    rewards = np.asarray(mean_rewards)
    window = min(100, len(rewards))
    if window > 0:
        rolling = np.convolve(rewards, np.ones(window) / window, mode="valid")
    else:
        rolling = np.array([])

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(
        np.arange(1, len(rewards) + 1),
        rewards,
        color="#b8b8b4",
        linewidth=0.8,
        label="Episode mean score (20 agents)",
    )
    ax.plot(
        np.arange(window, window + len(rolling)),
        rolling,
        color="#1f77b4",
        linewidth=2.0,
        label="{}-episode mean".format(window),
    )
    ax.axhline(30, color="#d62728", linewidth=1.0, linestyle="--",
               label="solved (+30)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Mean score")
    ax.set_title("PPO without critic/GAE on Reacher (20 agents)")
    ax.grid(True, color="#e6e6e2", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("Saved score plot to", output_path)


def main(args=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=1200,
                        help="1200 is the run that reaches the +30 bar")
    parser.add_argument("--tmax", type=int, default=1001,
                        help="steps per rollout; 1001 is a full Reacher episode")
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--epsilon", type=float, default=0.1,
                        help="probability ratio hyperparameter")
    parser.add_argument("--epsilon-decay", type=float, default=0.999,
                        help="per-episode multiplier on epsilon; 1.0 keeps it "
                             "fixed, which is what the SPO paper does")
    parser.add_argument("--sgd-epoch", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--lr-step", type=float, default=1e-4,
                        help="subtracted from lr every --lr-step-every "
                             "episodes; 0 keeps lr fixed")
    parser.add_argument("--lr-step-every", type=int, default=400,
                        help="episodes between lr decrements")
    parser.add_argument("--lr-min", type=float, default=1e-4,
                        help="lr never goes below this")
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--std0", type=float, default=0.5,
                        help="initial action standard deviation")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--print-every", type=int, default=10)
    parser.add_argument("--binary", default=DEFAULT_BINARY)
    parser.add_argument("--worker-id", type=int, default=0,
                        help="bump this to run two trainings side by side")
    parser.add_argument("--checkpoint", default="reacher_ppo_nocritic.policy")
    parser.add_argument("--plot", default="reacher_ppo_nocritic_scores.png")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--graphics", action="store_true",
                        help="open the Unity window instead of running headless")
    parsed = parser.parse_args(args)

    envs = ReacherEnv(
        file_name=parsed.binary,
        seed=parsed.seed,
        no_graphics=not parsed.graphics,
        worker_id=parsed.worker_id,
    )

    try:
        train(
            envs,
            episodes=parsed.episodes,
            tmax=parsed.tmax,
            discount_rate=parsed.discount,
            epsilon=parsed.epsilon,
            epsilon_decay=parsed.epsilon_decay,
            sgd_epoch=parsed.sgd_epoch,
            lr=parsed.lr,
            lr_step=parsed.lr_step,
            lr_step_every=parsed.lr_step_every,
            lr_min=parsed.lr_min,
            hidden=parsed.hidden,
            std0=parsed.std0,
            seed=parsed.seed,
            print_every=parsed.print_every,
            checkpoint_path=parsed.checkpoint,
            plot_path=parsed.plot,
            resume_from=parsed.checkpoint if parsed.resume else None,
        )
    finally:
        envs.close()


if __name__ == "__main__":
    main()
