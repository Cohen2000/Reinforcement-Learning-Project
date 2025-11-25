import time
import numpy as np
import random
import os
import torch
import json
import hockey.hockey_env as h_env
from src.algorithms.sac.agent import SACAgent

def train_episode(env, agent, max_steps=250, show=False):
    weak_opponent = random.random() < 0.5
    opponent = h_env.BasicOpponent(weak=weak_opponent)
    opponent_type = "weak" if weak_opponent else "strong"
    print(f"Training Episode: opponent = {opponent_type}")

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

        close_bonus = info.get("reward_closeness_to_puck", 0)
        touch_bonus = info.get("reward_touch_puck", 0)
        step_penalty = -0.01 * step

        win_bonus = 0.0
        if done:
            winner = info.get("winner", 0)
            if winner == 1:
                win_bonus = 30.0
                win_flag = 1
                result = "win"
            elif winner == -1:
                win_bonus = -5.0
                result = "loss"
            else:
                win_bonus = -10.0
                result = "draw"

        shaped_reward = reward + 5 * close_bonus + 2 * touch_bonus + step_penalty + win_bonus
        ep_reward += shaped_reward

        agent.store_transition((state, act_agent, shaped_reward, next_state, done))
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

def run_training(alpha, gamma, episodes=15000, log_interval=500, test_games=500):
    print(f"Start experiment: Alpha={alpha}, Gamma={gamma}, Episods={episodes}")

    sac_config = {
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "gamma": gamma,
    "tau": 0.005,
    "batch_size": 128,
    "auto_entropy": True if alpha == "auto" else False,
    "alpha": 0.2 if alpha == "auto" or alpha == "0.2" else (0.4 if alpha == "0.4" else 0.0),
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
            print(f"[Alpha={alpha}, Gamma={gamma}] After Episode {ep}: Winrate = {win_rate:.2f}%")

    model_path = os.path.join("models", "sac", "hockey", f"final_model_alpha_{alpha}_gamma_{gamma}.pt")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    agent.save(model_path)
    print(f"Finale Agent saved at: {model_path}")

    exp_folder = os.path.join('src', 'algorithms', 'sac', 'experiments')
    os.makedirs(exp_folder, exist_ok=True)
    result_file = os.path.join(exp_folder, f"alpha_{alpha}_gamma_{gamma}.json")
    results = {
        "win_rate_points": win_rate_points,
        "rewards_history": rewards_history,
        "steps_history": steps_history
    }
    with open(result_file, 'w') as f:
        json.dump(results, f)
    print(f"Results saved at: {result_file}")

    env.close()
    return win_rate_points

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha', type=str, choices=["auto", "0", "0.2"], required=True,
                        help="Alpha-Wert (auto, 0 oder 0.2)")
    parser.add_argument('--gamma', type=float, choices=[0.90, 0.95, 0.99], required=True,
                        help="Gamma-Wert")
    args = parser.parse_args()
    
    run_training(args.alpha, args.gamma)
