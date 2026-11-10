# Reinforcement Learning for Hockey

A Python/PyTorch project exploring continuous-control agents in the [Hockey environment](https://github.com/martius-lab/hockey-env). The repository contains TD3 and SAC implementations and experiments with Random Network Distillation (RND) for exploration.

## Start here

- [src/algorithms/](src/algorithms/): algorithm implementations and training/evaluation scripts.
- [src/main.py](src/main.py): command-line entry point and available arguments.
- [src/](src/): also contains saved model checkpoints.

**Focus:** reinforcement-learning training loops, actor-critic methods, exploration and comparing agent behaviour.

## Scope

The main CLI is only partially wired: SAC training is restricted to `Hockey-One-v0`, while the TD3 branches and SAC test branch are placeholders. Inspect the algorithm-specific scripts before running experiments. The CLI also lists Pendulum, LunarLander and HalfCheetah environments, but this is not evidence that every algorithm/environment combination runs through that entry point.

## Setup

From the repository directory:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 -m pip install git+https://github.com/martius-lab/hockey-env.git
```

Use the installed environment to inspect the CLI:

```bash
python src/main.py --help
```

Training requires the environment dependencies and the configuration expected by the chosen script. Existing checkpoints are experiment artifacts; no general benchmark performance claim is made here.
