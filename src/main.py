# src/main.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import argparse
import gymnasium as gym
import hockey.hockey_env as h_env  
from src.algorithms.sac.hockey_train import train_hockey 
from algorithms.rnd.rnd_train_hockey import train as train_rnd_hockey
from algorithms.rnd.rnd_test_hockey import test as test_rnd_hockey
from hockey.hockey_env import Mode

SUPPORTED_ALGORITHMS = ["td3", "sac", "rnd"]
SUPPORTED_ENVIRONMENTS = [
    "Hockey-One-v0",
    "Pendulum-v1",
    "LunarLander-v2",
    "LunarLanderContinuous-v2",
    "HalfCheetah-v4",
]

def main():
    parser = argparse.ArgumentParser(description="Train or test a RL agent.")
    parser.add_argument("--algo", type=str, required=True, help="Algorithm to use.")
    parser.add_argument("--env", type=str, required=True, help="Environment to use.")
    parser.add_argument("--mode", type=str, choices=["train", "test"], default="train", help="Mode: train or test.")
    parser.add_argument("--episodes", type=int, default=20000, help="Number of episodes.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to an existing model to continue training or for testing.")
    parser.add_argument("--stats", type=str, default=None, help="Path to an existing stats file to continue tracking performance.")
    parser.add_argument("--render", action="store_true", help="Render the environment (only recommended for testing).")
   
    args = parser.parse_args()

    if args.algo not in SUPPORTED_ALGORITHMS:
        raise ValueError(f"Unsupported algorithm: {args.algo}. Supported: {SUPPORTED_ALGORITHMS}")
    if args.env not in SUPPORTED_ENVIRONMENTS:
        raise ValueError(f"Unsupported environment: {args.env}. Supported: {SUPPORTED_ENVIRONMENTS}")

    render_mode = "human" if args.mode == "test" else None

    if args.env == "Hockey-One-v0":
        env = h_env.HockeyEnv()
    else:
        env = gym.make(args.env, render_mode=render_mode)

    if args.mode == "train":
        if args.algo == "sac":
            if args.env == "Hockey-One-v0":
                train_hockey(env, episodes=args.episodes, show=args.render)
            else:
                raise ValueError("SAC is currently implemented only for 'Hockey-One-v0'.")
        elif args.algo == "td3":
            print("TD3 training has not been modified.")
        elif args.algo == "rnd":
            train_rnd_hockey(env, args.episodes, args.env)
    elif args.mode == "test":
        if args.algo == "sac":
            print("Not in use")
            #test_sac_hockey(env, model_path, episodes=args.episodes, render=args.render)
        elif args.algo == "td3":
            print("TD3 test has not been modified.")
        elif args.algo == "rnd":
            test_rnd_hockey(env, args.episodes, args.env)

    
    env.close()

if __name__ == "__main__":
    main()
