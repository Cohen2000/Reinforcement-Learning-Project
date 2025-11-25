import argparse
import imageio
import numpy as np
import torch
import hockey.hockey_env as h_env
from src.algorithms.sac.agent import SACAgent
import random
import os

def record_gif(model_path: str, output_gif: str, episodes: int = 3):
    env = h_env.HockeyEnv()
    obs_size = env.observation_space.shape[0]
    
    config = {
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        "gamma": 0.95,
        "tau": 0.005,
        "batch_size": 128,
        "auto_entropy": True,
        "alpha": 0.2,
        "actor_lr": 1e-4,
        "critic_lr": 1e-3,
        "alpha_lr": 1e-4,
        "replay_capacity": 100000,
        "grad_steps": 32
    }
    
    agent = SACAgent(obs_size, env.action_space, config)
    agent.load(model_path)
    print(f"Model loaded from {model_path}")
    opponent = h_env.BasicOpponent(weak=True)
    
    frames = []
    for ep in range(episodes):
        reset_ret = env.reset()
        state = reset_ret[0] if isinstance(reset_ret, tuple) else reset_ret
        done = False
        while not done:
            frame = env.render(mode='rgb_array')
            frames.append(frame)
            
            action_agent = agent.act(state, eval_mode=True)
            
            if hasattr(env, "obs_agent_two"):
                opp_state = env.obs_agent_two()
            else:
                opp_state = None
            action_opp = opponent.act(opp_state)
            
            full_action = np.concatenate([action_agent, action_opp])
            
            step_result = env.step(full_action)
            if len(step_result) == 5:
                next_state, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                next_state, reward, done, info = step_result
            
            state = next_state
            
    env.close()
    
    imageio.mimsave(output_gif, frames, fps=30)
    print(f"GIF saved to {output_gif}")

def main():
    parser = argparse.ArgumentParser(description="Record a gameplay demo as a GIF using a trained SAC model.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the saved SAC model")
    parser.add_argument("--output_gif", type=str, default="demo.gif", help="Filename for the output GIF")
    parser.add_argument("--episodes", type=int, default=3, help="Number of episodes to record")
    args = parser.parse_args()
    
    record_gif(args.model_path, args.output_gif, args.episodes)

if __name__ == "__main__":
    main()
