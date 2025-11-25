import torch
import random
import string
import pickle
import numpy as np
import matplotlib.pyplot as plt
from .rnd_agent import PPO, Memory
import hockey.hockey_env as h_env 

#for TD3 agent
from ..td3.td3_versions import td3_pink_noisy as td3_pink_noisy

def train(env, episodes, env_name, load_model = True, render_train = False, train_itself = True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]       
    max_timesteps = 300 #max timesteps in one episode
    n_latent_var = 64
    update_timestep = 5000 #update policy every n timesteps
    lr = 0.0001
    betas = (0.9, 0.999)
    gamma = 0.99
    K_epochs = 10
    eps_clip = 0.15 #clip parameter for PPO
    rnd_beta = 0.1 #intrinsic reward scaling factor initialization

    memory = Memory()
    agent = PPO(state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip, rnd_beta)
    player2_weak = h_env.BasicOpponent(weak=True)
    player2_strong = h_env.BasicOpponent(weak=False)

    if load_model:
        load_path = f"models/rnd/rnd_{env_name}.pt"
        checkpoint = torch.load(load_path)
        agent.policy.load_state_dict(checkpoint['policy_state_dict'])
        agent.policy_old.load_state_dict(checkpoint['policy_state_dict'])

        if train_itself: #If training contains the agent itself as an enemy
            load_path = f"models/rnd/rnd_{env_name}_player2.pt"
            checkpoint2 = torch.load(load_path)
            memory_player_2 = Memory()
            player2_own = PPO(state_dim, action_dim, n_latent_var, 0.0, betas, gamma, K_epochs, eps_clip, rnd_beta)
            player2_own.policy.load_state_dict(checkpoint2['policy_state_dict'])

            #player 4 to train against an even older state of the agent
            load_path4 = f"models/rnd/rnd_{env_name}_player4.pt"
            checkpoint4 = torch.load(load_path4)
            player4_own = PPO(state_dim, action_dim, n_latent_var, 0.0, betas, gamma, K_epochs, eps_clip, rnd_beta)
            player4_own.policy.load_state_dict(checkpoint4['policy_state_dict'])

            #code to load in the TD3 agent
            player_td3 = td3_pink_noisy.TD3(env.observation_space, 
                        env.action_space, 
                        env.num_actions, 
                        env.max_timesteps,
                        eps = 0.1, #opts default
                        learning_rate_actor = 1e-4, #opts default 
                        learning_rate_critic = 1e-4) #opts default

            player3_path = f"models/rnd/TD3_Hockey_episode40000.pth"
            player_td3.restore_state(torch.load(player3_path, map_location=torch.device(device)))

            player2s = [player2_strong, player_td3, player2_own, player4_own, player2_strong, player2_weak, player_td3, player2_own, player2_strong, player2_weak, player_td3]
            #how many times the enemy own-player is updated
            player2_update_times = 0

    avg_rewards = []
    avg_lengths = []
    avg_winrates = []
    avg_intrinsic = []

    rewards = []
    lengths = []
    win_stats = []
    timestep = 0
    print(f"Training RND for {episodes} episodes.")

    for i_episode in range(1, episodes+1):
        #logic to alternate enemies
        if train_itself:
            player2 = player2s[i_episode % 11]
        else:
            player2 = player2_weak if i_episode % 3 == 0 else player2_strong

        state, _info = env.reset()
        obs_agent2 = env.obs_agent_two()

        #beta scaling with episodes
        agent.rnd_beta = max(0.01, 0.2 * (1 - i_episode / episodes))

        running_reward = 0
        for t in range(max_timesteps):
            timestep += 1

            if render_train:
                env.render(mode="human")

            action = agent.policy_old.act(state, memory)
            discrete_action = env.discrete_to_continous_action(action)

            if not train_itself:
                action_player_2 = player2.act(obs_agent2)
            else:
                if i_episode % 11 == 2 or i_episode % 11 == 3 or i_episode % 11 == 7:
                    action_player_2 = env.discrete_to_continous_action(player2.policy_old.act(obs_agent2, memory_player_2))
                else:
                    action_player_2 = player2.act(obs_agent2)

            full_action = np.hstack([discrete_action, action_player_2])
            state, reward, done, _trunc, _info = env.step(full_action)

            #save reward and is_terminal:
            memory.rewards.append(reward)
            memory.is_terminals.append(done)

            if train_itself:
                #only update the memory where the agent actually has one (own)
                if i_episode % 11 == 2 or i_episode % 11 == 3 or i_episode % 11 == 7:
                    if player2_update_times < 2:
                        memory_player_2.rewards.append(-reward)
                        memory_player_2.is_terminals.append(done)

            #update period
            if timestep % update_timestep == 0:
                agent.update(memory)
                memory.clear_memory()

                if train_itself:
                    if i_episode % 11 == 2 or i_episode % 11 == 3 or i_episode % 11 == 7:
                        if player2_update_times < 2:
                            player2_own.update(memory_player_2)
                            memory_player_2.clear_memory()
                            player2_update_times += 1

                timestep = 0

            obs_agent2 = env.obs_agent_two()
            
            running_reward += reward
            if done or _trunc:
                win_stats.append(_info["winner"])
                break
        rewards.append(running_reward)
        lengths.append(t)

        #every 500 episodes print the current mean reward and track statistics
        if i_episode % 500 == 0:
            avg_reward = np.mean(rewards[-500:])
            avg_rewards.append(avg_reward)

            avg_length = np.mean(lengths[-500:])
            avg_lengths.append(avg_length)

            current_wins = win_stats[-500:].count(1)
            avg_winrate = round(current_wins / 500 * 100, 2)
            avg_winrates.append(avg_winrate)

            if len(agent.intrinsic_rewards) > 0:
                avg_intrin = np.mean(agent.intrinsic_rewards[-500:])
            else:
                avg_intrin = 0.0

            avg_intrinsic.append(avg_intrin)

            print('Episode {} \t rewards: {} \t lengths: {} \t winrate: {} \t intrinsic: {}'.format(i_episode, avg_reward, avg_length, avg_winrate, avg_intrin))

    win_loss_draw_rates = calculate_outcomes(win_stats)
    total_reward_avg = np.mean(rewards)
    print(f"Win percentage: {win_loss_draw_rates['win_percent']}%")
    print(f"Draw percentage: {win_loss_draw_rates['draw_percent']}%")
    print(f"Loss percentage: {win_loss_draw_rates['loss_percent']}%")
    print(f"Average reward over entire training: {total_reward_avg}")

    #save model
    suffix = ''.join(random.choices(string.ascii_letters + string.digits, k=4))
    model_path = f"models/rnd/rnd_{env_name}_{suffix}.pt"
    checkpoint_save = {
        'policy_state_dict': agent.policy.state_dict()
    }
    torch.save(checkpoint_save, model_path)
    print(f"Model saved at {model_path}")

    #accumulate statistics
    stats = {
        'extrinsic': rewards,
        'intrinsic': agent.intrinsic_rewards,
        'lengths': lengths,
        'win_rates': win_stats,
        'episodes': episodes
    }
    stats_path = f"models/rnd/stats/rnd_{env_name}_{suffix}.pkl"
    with open(stats_path, 'wb') as f:
        pickle.dump(stats, f)

    img_path = f"models/rnd/img/rnd_{env_name}_{suffix}.png"    

    plot_training_results(rewards, agent.intrinsic_rewards, win_stats, lengths, img_path)

def calculate_outcomes(results):
    wins = results.count(1)
    draws = results.count(0)
    losses = results.count(-1)
    
    total = len(results)

    return {
        "win_percent": round(wins / total * 100, 2),
        "draw_percent": round(draws / total * 100, 2),
        "loss_percent": round(losses / total * 100, 2)
    }


def plot_training_results(extrinsic, intrinsic, win_rates, lengths, save_path):
    window = 500
    plt.figure(figsize=(15, 10))
    
    #smoothed Extrinsic Reward
    plt.subplot(2, 2, 1)
    smooth_extrinsic = [np.mean(extrinsic[max(0,i-window):i+1]) for i in range(len(extrinsic))]
    plt.plot(smooth_extrinsic)
    plt.title(f'Extrinsic Reward (Smoothed, window={window})')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    
    #intrinsic Reward per update
    plt.subplot(2, 2, 2)
    x_intr = np.arange(len(intrinsic))
    smooth_intrinsic = [np.mean(intrinsic[max(0, i-window):i+1]) for i in range(len(intrinsic))]
    plt.plot(x_intr, smooth_intrinsic)
    plt.title(f'Intrinsic Reward (Smoothed, window={window})')
    plt.xlabel('Update Iteration')
    plt.ylabel('Intrinsic Reward')
    
    #episode Length
    plt.subplot(2, 2, 3)
    smooth_lengths = [np.mean(lengths[max(0,i-window):i+1]) for i in range(len(lengths))]
    plt.plot(smooth_lengths)
    plt.title(f'Episode Length (Smoothed, window={window})')
    plt.xlabel('Episode')
    plt.ylabel('Steps')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show() 
