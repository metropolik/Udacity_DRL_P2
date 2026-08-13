"""Watch a trained checkpoint drive the 20 Reacher arms, in the Unity window.

    python watch_reacher.py                      # the solving run, greedy, loops forever
    python watch_reacher.py --sample             # the STOCHASTIC policy - what the
                                                 # training curve actually plots
    python watch_reacher.py --random             # a random policy, for scale
    python watch_reacher.py --episodes 3         # then stop
    python watch_reacher.py --headless           # numbers only, ~20x faster

Ctrl-C at any point; the Unity process is closed cleanly.

Why `--sample` matters here
===========================
`reacher_ppo_nocritic.py --play` evaluates the MEAN action. The score printed
during training is the SAMPLED action, with the policy's own std (~0.37 at the
end of run D). Those are different policies and they score differently, so if
you are chasing a wobble in the training curve, `--sample` is the one that
reproduces it.

The live line
=============
    ep 1 | step  480/1001 | score 15.42 | arms on target 17/20 | 62% of steps

`arms on target` is how many of the 20 hands are inside the goal sphere right
now (Unity pays 0.01-0.04 for exactly that), and the trailing percentage is the
share of all steps so far in which the average arm was on target. A solved
policy sits near 20/20 once the arms have caught up with their spheres, and the
score is just that percentage times ~33.
"""

import argparse
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from reacher_ppo_nocritic import (  # noqa: E402
    DEFAULT_BINARY,
    Policy,
    ReacherEnv,
    device,
)

DEFAULT_CHECKPOINT = os.path.join(HERE, "runs", "D_noobsnorm_lr3e-4.policy")


def load_policy(checkpoint_path, state_size, action_size):
    """Rebuild the policy from a checkpoint, inferring the hidden width from it.

    Older checkpoints also carry an `obs_norm` entry from the observation
    normalizer that used to be in the training script; it is ignored.
    """
    chkpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    hidden = chkpt["policy"]["fc_net.fc1.weight"].shape[0]

    policy = Policy(state_size=state_size, action_size=action_size,
                    hidden=hidden).to(device)
    policy.load_state_dict(chkpt["policy"])
    policy.eval()

    std = torch.exp(torch.clamp(policy.log_std, policy.log_std_min,
                                policy.log_std_max)).mean().item()
    print("checkpoint : {}".format(checkpoint_path))
    print("hidden     : {}   action std: {:.3f}".format(hidden, std))
    return policy


def run_episode(envs, policy, mode, tmax, refresh, train_mode):
    """One episode. `mode` is 'greedy', 'sample' or 'random'. Returns scores.

    `train_mode` is Unity's time scale, not ours: False runs the physics at
    real time so you can watch it, True runs it as fast as it will go.
    """
    states = envs.reset(train_mode=train_mode)
    scores = np.zeros(envs.num_envs)
    on_target_steps = 0
    t0 = time.time()

    for t in range(tmax):
        if mode == "random":
            actions = np.random.uniform(-1, 1, (envs.num_envs, envs.action_size))
        else:
            batch = torch.as_tensor(states, dtype=torch.float32, device=device)
            with torch.no_grad():
                if mode == "greedy":
                    actions, _ = policy(batch)
                    actions = actions.cpu().numpy()
                else:
                    actions, _ = policy.act(batch)

        next_states, rewards, dones = envs.step(actions)
        scores += rewards
        states = next_states

        on_now = int((rewards > 0).sum())
        on_target_steps += on_now

        if (t + 1) % refresh == 0 or t + 1 == tmax:
            sys.stdout.write(
                "\r  step {:4d}/{:d} | score {:6.2f} | arms on target {:2d}/{:d}"
                " | {:3.0f}% of steps ".format(
                    t + 1, tmax, scores.mean(), on_now, envs.num_envs,
                    100.0 * on_target_steps / ((t + 1) * envs.num_envs),
                )
            )
            sys.stdout.flush()

        if dones.any():
            break

    sys.stdout.write("\r" + " " * 78 + "\r")
    print("  score {:6.2f}   [per-arm min {:.2f}, max {:.2f}]   {:.0f}s".format(
        scores.mean(), scores.min(), scores.max(), time.time() - t0))
    return scores


def main(args=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--binary", default=DEFAULT_BINARY)
    parser.add_argument("--episodes", type=int, default=0,
                        help="0 = loop until Ctrl-C")
    parser.add_argument("--tmax", type=int, default=1001)
    parser.add_argument("--sample", action="store_true",
                        help="sample from the policy instead of using its mean")
    parser.add_argument("--random", action="store_true",
                        help="ignore the checkpoint, act uniformly at random")
    parser.add_argument("--headless", action="store_true",
                        help="no Unity window; much faster, numbers only")
    parser.add_argument("--refresh", type=int, default=20,
                        help="steps between live-line updates")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--worker-id", type=int, default=0)
    parsed = parser.parse_args(args)

    mode = "random" if parsed.random else ("sample" if parsed.sample else "greedy")

    envs = ReacherEnv(
        file_name=parsed.binary,
        seed=parsed.seed,
        no_graphics=parsed.headless,
        worker_id=parsed.worker_id,
    )

    try:
        policy = None
        if mode != "random":
            policy = load_policy(
                parsed.checkpoint, envs.state_size, envs.action_size)

        print("mode       : {}\n".format(mode))

        per_episode = []
        e = 0
        while parsed.episodes == 0 or e < parsed.episodes:
            e += 1
            print("episode {}".format(e))
            scores = run_episode(envs, policy, mode, parsed.tmax,
                                 parsed.refresh, train_mode=parsed.headless)
            per_episode.append(scores.mean())
            print("  running mean over {} episode(s): {:.2f}\n".format(
                len(per_episode), float(np.mean(per_episode))))
    except KeyboardInterrupt:
        print("\nstopped.")
    finally:
        envs.close()


if __name__ == "__main__":
    main()
