#!/usr/bin/env python3
import time
import numpy as np
import random
import os
import torch
import json
import argparse
import hockey.hockey_env as h_env
from src.algorithms.sac.agent import SACAgent

def train_episode(env, agent, max_steps=250, show=False):
    weak_opponent = random.random() < 0.5
    opponent = h_env.BasicOpponent(weak=weak_opponent)
    opponent_type = "weak" if weak_opponent else "strong"
    print(f"Training Episode (No Reward Shaping): Opponent = {opponent_type}")

    reset_ret = env.reset()
    state = reset_ret[0] if isinstance(reset_ret, tuple) else reset_ret

    ep_reward = 0.0
    ep_steps = 0
    win_flag = 0
    result = "Draw"  

    for step in range(max_steps):
        ep_steps += 1
        act_agent = agent.act(state)
        if hasattr(env, "obs_agent_two"):
            opp_state = env.obs_agent_two()
        else:
            opp_state = None
        act_opp = opponent.act(opp_state)
        full_action = np.concatenate([act_agent, act_opp])
        
        step_result = env.step(full_action)
        if len(step_result) == 5:
            next_state, reward, terminated, truncated, info = step_result
            done = terminated or truncated
        else:
            next_state, reward, done, info = step_result

        raw_reward = reward

        ep_reward += raw_reward

        agent.store_transition((state, act_agent, raw_reward, next_state, done))
        state = next_state

        if done:
            break

    return ep_reward, ep_steps, win_flag, result

def evaluate_agent(env, agent, test_games=500, max_steps=250):
    wins = 0
    for _ in range(test_games):
        opponent = h_env.BasicOpponent(weak=True)
        reset_ret = env.reset()
        state = reset_ret[0] if isinstance(reset_ret, tuple) else reset_ret

        done = False
        while not done:
            action = agent.act(state, eval_mode=True)
            if hasattr(env, "obs_agent_two"):
                opp_state = env.obs_agent_two()
            else:
                opp_state = None
            opp_action = opponent.act(opp_state)
            full_action = np.concatenate([action, opp_action])
            
            step_result = env.step(full_action)
            if len(step_result) == 5:
                next_state, reward, terminated, truncated, _ = step_result
                done = terminated or truncated
            else:
                next_state, reward, done, _ = step_result
            state = next_state

        if hasattr(env, "winner") and env.winner == 1:
            wins += 1

    win_rate = wins / test_games * 100
    return win_rate

def run_training(alpha, gamma, episodes=10000, log_interval=500, test_games=500):
    print(f"Starting training experiment (No Reward Shaping): Alpha={alpha}, Gamma={gamma}, Episodes={episodes}")

    sac_config = {
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        "gamma": gamma,
        "tau": 0.005,
        "batch_size": 128,
        "auto_entropy": True if alpha == "auto" else False,
        "alpha": 0.2 if alpha == "auto" else 0.0,
        "actor_lr": 1e-4,
        "critic_lr": 1e-3,
        "alpha_lr": 1e-4,
        "replay_capacity": 100000,
        "max_steps": 250,
        "grad_steps": 32
    }

    env = h_env.HockeyEnv()
    obs_size = env.observation_space.shape[0]
    agent = SACAgent(obs_size, env.action_space, sac_config)

    win_rate_points = []
    rewards_history = []
    steps_history = []

    for ep in range(1, episodes + 1):
        ep_reward, ep_steps, win_flag, result = train_episode(env, agent, max_steps=sac_config["max_steps"], show=False)
        rewards_history.append(ep_reward)
        steps_history.append(ep_steps)
        print(f"Episode {ep}/{episodes} - Reward: {ep_reward:.2f} - Steps: {ep_steps} - Result: {result}")

        for _ in range(sac_config["grad_steps"]):
            agent.update()

        if ep % log_interval == 0:
            win_rate = evaluate_agent(env, agent, test_games=test_games, max_steps=sac_config["max_steps"])
            win_rate_points.append((ep, win_rate))
            print(f"[No Reward Shaping] [Alpha={alpha}, Gamma={gamma}] After Episode {ep}: Win Rate = {win_rate:.2f}%")

    model_path = os.path.join("models", "sac", "hockey", f"final_model_no_shaping_alpha_{alpha}_gamma_{gamma}.pt")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    agent.save(model_path)
    print(f"Final agent saved to: {model_path}")

    exp_folder = os.path.join('src', 'algorithms', 'sac', 'experiments')
    os.makedirs(exp_folder, exist_ok=True)
    result_file = os.path.join(exp_folder, f"no_shaping_pp_alpha_{alpha}_gamma_{gamma}.json")
    results = {
        "win_rate_points": win_rate_points,
        "rewards_history": rewards_history,
        "steps_history": steps_history
    }
    with open(result_file, 'w') as f:
        json.dump(results, f)
    print(f"Results saved to: {result_file}")

    env.close()
    return win_rate_points

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha', type=str, choices=["auto", "0"], required=True,
                        help="Alpha value (choose 'auto' or '0')")
    parser.add_argument('--gamma', type=float, choices=[0.90, 0.95], required=True,
                        help="Gamma value (choose 0.90 or 0.95)")
    args = parser.parse_args()
    
    run_training(args.alpha, args.gamma)
