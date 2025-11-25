import os
import time
import random
import numpy as np
import torch
import hockey.hockey_env as h_env
from src.algorithms.sac.agent import SACAgent

def update_self_play_opponent(agent, obs_size, env_action_space, sac_config):
    opponent = SACAgent(obs_size, env_action_space, sac_config)
    opponent.actor.load_state_dict(agent.actor.state_dict())
    opponent.actor.eval()
    return opponent

def train_hockey(env, episodes=80000, log_interval=500, show=False):
    sac_config = {
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
        "max_steps": 250,
        "grad_steps": 32
    }
    obs_size = env.observation_space.shape[0]
    agent = SACAgent(obs_size, env.action_space, sac_config)
    
    #project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
    script_dir = os.path.dirname(__file__)
    final_model_path = os.path.join(script_dir, "../../final_model_selfplay.pt")

    
    self_play_opponent = update_self_play_opponent(agent, obs_size, env.action_space, sac_config)
    update_interval = 5000  
    

    episode_log_data = []
    rewards_history = []
    steps_history = []
    
    def select_opponent(ep):
        if ep < 20000:
            if random.random() < 0.5:
                return h_env.BasicOpponent(weak=True), "bot_weak"
            else:
                return h_env.BasicOpponent(weak=False), "bot_strong"
        elif ep < 40000:
            if random.random() < 0.5:
                return self_play_opponent, "selfplay"
            else:
                if random.random() < 0.5:
                    return h_env.BasicOpponent(weak=True), "bot_weak"
                else:
                    return h_env.BasicOpponent(weak=False), "bot_strong"
        else:
            if random.random() < 0.75:
                return self_play_opponent, "selfplay"
            else:
                if random.random() < 0.5:
                    return h_env.BasicOpponent(weak=True), "bot_weak"
                else:
                    return h_env.BasicOpponent(weak=False), "bot_strong"
    
    for ep in range(1, episodes + 1):
        if ep % update_interval == 0:
            self_play_opponent = update_self_play_opponent(agent, obs_size, env.action_space, sac_config)
            print(f"Self-play opponent updated at episode {ep}.")
        
        opponent, opp_type = select_opponent(ep)
        reset_ret = env.reset()
        state = reset_ret[0] if isinstance(reset_ret, tuple) else reset_ret
        opp_state = env.obs_agent_two() if hasattr(env, "obs_agent_two") else None
        
        ep_reward = 0
        ep_steps = 0
        win_flag = 0
        info = {}
        
        for step in range(sac_config["max_steps"]):
            ep_steps += 1
            act_agent = agent.act(state)
            if opp_type == "selfplay":
                act_opp = opponent.act(opp_state, eval_mode=True)
            else:
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
            win_bonus = 30.0 if done and info.get("winner", 0) == 1 else (
                        -5.0 if done and info.get("winner", 0) == -1 else (-10.0 if done else 0.0))
            if done and info.get("winner", 0) == 1:
                win_flag = 1
            
            shaped_reward = reward + 5 * close_bonus + 2 * touch_bonus + step_penalty + win_bonus
            ep_reward += shaped_reward
            agent.store_transition((state, act_agent, shaped_reward, next_state, done))
            if show:
                time.sleep(0.01)
                env.render()
            state = next_state
            if hasattr(env, "obs_agent_two"):
                opp_state = env.obs_agent_two()
            if done:
                break
        
        rewards_history.append(ep_reward)
        steps_history.append(ep_steps)
        
        outcome_str = "Win" if win_flag == 1 else "Loss"
        print(f"Episode {ep}/{episodes} | Opponent: {opp_type} | Reward: {ep_reward:.2f} | Steps: {ep_steps} | Outcome: {outcome_str}")
        
        episode_log_data.append({"opp_type": opp_type, "win": win_flag})
        
        for _ in range(sac_config["grad_steps"]):
            agent.update()
        
        if ep % log_interval == 0:
            window_data = episode_log_data[-log_interval:]
            total_window = len(window_data)
            overall_wins = sum(item["win"] for item in window_data)
            overall_win_rate = (overall_wins / total_window * 100) if total_window > 0 else 0.0
            print(f"\nCheckpoint: Episode {ep}")
            print(f"  Avg Reward (last {total_window} episodes): {np.mean(rewards_history[-log_interval:]):.2f}")
            print(f"  Avg Steps (last {total_window} episodes): {np.mean(steps_history[-log_interval:]):.1f}")
            print(f"  Overall Win Rate (last {total_window} episodes): {overall_win_rate:.1f}%")
            opp_types = set(item["opp_type"] for item in window_data)
            for ot in opp_types:
                ot_data = [item for item in window_data if item["opp_type"] == ot]
                count = len(ot_data)
                wins = sum(item["win"] for item in ot_data)
                win_rate = (wins / count * 100) if count > 0 else 0.0
                print(f"    {ot}: {win_rate:.1f}% over {count} episodes")
            print()
    
    #final_model_path = os.path.join(project_root, "models", "sac", "hockey", "final_model_selfplay.pt")
    os.makedirs(os.path.dirname(final_model_path), exist_ok=True)
    agent.save(final_model_path)
    print(f"Training complete. Model saved to {final_model_path}")
    return rewards_history, steps_history, episode_log_data
