import matplotlib.pyplot as plt
import os
import pickle
import numpy as np
import seaborn as sns
import pandas as pd
import glasbey

def plot_fct(data, title = "", x_label = "", y_label = ""):
    plt.plot(data)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)

def plot_mean_std_sns(data, label, x_label, y_label, title, loc='upper right', window_size=20, show_fill = True):
    values = data

    mean_values = np.convolve(values, np.ones(window_size)/window_size, mode='valid')
    std_values = np.array([
        np.std(values[i:i+window_size]) for i in range(len(values) - window_size + 1)
    ])

    time = np.arange(len(mean_values))

    df = pd.DataFrame({
        x_label: np.arange(len(mean_values)),
        "Mean": mean_values,
        "Std": std_values
    })

    sns.lineplot(data=df, x=x_label, y="Mean", label=label)
    if show_fill:
        plt.fill_between(df[x_label], df["Mean"] - df["Std"], df["Mean"] + df["Std"], alpha=0.3, rasterized=True)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend(loc=loc)


def plot_violin_sns(data, label, x_label, y_label):
    df = pd.DataFrame({
        'Value': data,
        'Category': [label]*len(data)
    })
    sns.violinplot(
        data=df,
        x='Category',
        y='Value',
        cut=0,         
        inner='box'    
    )

    plt.xlabel(x_label)
    plt.ylabel(y_label)

def load_statistics(pkl_path):
    
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    print("Loaded statistics checkpoint from:", pkl_path)
    return data

def create_plots(pkl_path):
    data = load_statistics(pkl_path)
    
    rewards = data['rewards']
    cumulative_rewards = np.cumsum(rewards)
    cumulative_games = np.arange(1, len(rewards) + 1)
    cumulative_rewards = cumulative_rewards / cumulative_games

    win_draw_loose = np.array(data['win_draw_loose'])
    cumulative_wins = np.cumsum(win_draw_loose == 1)
    cumulative_games = np.arange(1, len(win_draw_loose) + 1)
    cumulative_win_rate = cumulative_wins / cumulative_games

    losses = np.array(data['losses'][1:])
    losses = np.reshape(losses, (losses.shape[0]*losses.shape[1], losses.shape[2]))
    critic_losses = losses[:, 0]
    actor_losses = losses[:, 1]

    lengths = np.array(data['lengths'])

    plot_mean_std_sns(rewards, "", "", "", "", 0, 200)
    plt.show()
    plot_mean_std_sns(critic_losses, "", "", "", "", 0, 200)
    plt.show()
    plot_mean_std_sns(actor_losses, "", "", "", "", 0, 200)
    plt.show()
    plot_fct(cumulative_win_rate)
    plt.show()
    plot_mean_std_sns(lengths, "", "", "", "", 0, 200)
    plt.show()


def overlaying_plots(files, labels, folder):
    os.makedirs(folder, exist_ok=True)
    data = load_pkl_files(files)
    palette = glasbey.create_palette(palette_size=4, colorblind_safe=True)

    sns.set_theme(style="whitegrid", font_scale=1.7, palette=palette)
    plt.figure(figsize=(10,9))
    for i, obj in enumerate(data):
        rewards = obj['rewards']
        plot_mean_std_sns(rewards, labels[i], "Training Episode (Number of Games played)", "Average Reward over one Game", None, loc='lower right', window_size=500, show_fill=False)
    export_plot(f"{folder}/rewards.pdf")
    export_plot_png(f"{folder}/rewards.png")


    sns.set_theme(style="whitegrid", font_scale=1.7, palette=palette)
    plt.figure(figsize=(10,9))
    for i, obj in enumerate(data):
        lengths = obj['lengths']
        plot_mean_std_sns(lengths, labels[i], "Training Episode (Number of Games played)", "Length of Game Until Win/Draw", None, window_size=500, show_fill=False)
    export_plot(f"{folder}/lengths.pdf")
    export_plot_png(f"{folder}/lengths.png")


    sns.set_theme(style="whitegrid", font_scale=1.7, palette=palette)
    plt.figure(figsize=(10,9))
    for i, obj in enumerate(data):
        losses_c = [entry[0] for entry in obj['losses']]
        losses_a = [entry[1] for entry in obj['losses']]
        losses = np.array(losses_c)
        losses = np.reshape(losses, (losses.shape[0]*losses.shape[1]))

        plot_mean_std_sns(losses, labels[i], "Training Episode (Number of Games played)", "Losses of Critic", None, window_size=500, show_fill=False)
    export_plot(f"{folder}/losses_critic.pdf")
    export_plot_png(f"{folder}/losses_critic.png")

    sns.set_theme(style="whitegrid", font_scale=1.7, palette=palette)
    plt.figure(figsize=(10,9))
    for i, obj in enumerate(data):
        losses_c = [entry[0] for entry in obj['losses']]
        losses_a = [entry[1] for entry in obj['losses']]
        losses_a = np.array([float(loss) for sublist in losses_a for loss in sublist])
        losses = np.array(losses_a)

        plot_mean_std_sns(losses, labels[i], "Training Episode (Number of Games played)", "Losses of Actor", None, window_size=500, show_fill=False)
    export_plot(f"{folder}/losses_actor.pdf")
    export_plot_png(f"{folder}/losses_actor.png")


def barplots(files, labels, folder, ax=None):
    palette = glasbey.create_block_palette([2, 2, 2, 2])
    sns.set_theme(style="whitegrid", font_scale=1.5, palette=palette)

    if ax is None:
        fig, ax = plt.subplots(figsize=(10,9))
    
    num_existing_bars = len(ax.patches) // 2
    bar_width = 0.3
    x_offset = num_existing_bars * 2 * bar_width

    for i, obj in enumerate(files):
        num_existing_bars = i
        bar_width = 0.2
        x_offset = num_existing_bars * 2 * (bar_width+0.03) 

        weak_winrate = obj[0]
        strong_winrate = obj[1]

        data = load_statistics(weak_winrate)
        win_draw_loose = np.array(data['win_draw_loose'])
        n_wins_weak = np.sum(win_draw_loose == 1) / len(win_draw_loose)

        data = load_statistics(strong_winrate)
        win_draw_loose = np.array(data['win_draw_loose'])
        n_wins_strong = np.sum(win_draw_loose == 1) / len(win_draw_loose)
        x_positions = np.array([0, bar_width]) + x_offset

        ax.bar(x_positions[0], n_wins_weak, width=bar_width, label=f"Weak - {labels[i]}")
        ax.bar(x_positions[1], n_wins_strong, width=bar_width, label=f"Strong - {labels[i]}")

        ax.text((x_positions[0] + x_positions[1]) / 2, -0.1, labels[i], ha="center", va="top", fontsize=17, fontweight='bold', transform=ax.transData)

    tick_positions = np.arange(num_existing_bars * 2 + 2) * (bar_width+0.03)
    tick_positions[1::2] -= 0.05
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(["Weak", "Strong"] * (num_existing_bars + 1), rotation=0)
    ax.set_ylabel("Winrate")

    plt.tight_layout()
    plt.savefig(f"{folder}/winrate.pdf", dpi=300, bbox_inches="tight")
    print(f"Plot successfully saved as {folder}/winrate.pdf")

    plt.savefig(f"{folder}/winrate.png", dpi=300, bbox_inches="tight")

    return ax 


def barplots_tournament(files, labels, folder, ax=None):
    palette = glasbey.create_block_palette([2, 2, 2, 2])
    sns.set_theme(style="whitegrid", font_scale=1.5)
    x=[0, 1, 2]
    if ax is None:
        fig, ax = plt.subplots(figsize=(7,6))
    
    num_existing_bars = len(ax.patches) // 2
    bar_width = 0.3
    x_offset = num_existing_bars * bar_width 

    for i, obj in enumerate(files):
        num_existing_bars = i
        bar_width = 0.2 
        x_offset = num_existing_bars * (bar_width+0.03) 

        winrate = obj

        data = load_statistics(winrate)
        win_draw_loose = np.array(data['win_draw_loose'])
        n_wins_weak = np.sum(win_draw_loose == 1) / len(win_draw_loose)

        ax.bar(i, n_wins_weak, label=labels[i])

    plt.xticks(x, labels)

    plt.tick_params(axis='x', length=0)
    plt.tick_params(axis='x', pad=10)
    ax.set_ylabel("Winrate")

    plt.tight_layout()
    plt.savefig(f"{folder}/winrate.pdf", dpi=300, bbox_inches="tight")
    print(f"Plot successfully saved as {folder}/winrate.pdf")

    return ax 


def overlaying_plots_tournament(files, labels, folder):
    os.makedirs(folder, exist_ok=True)
    data = load_pkl_files(files)
    palette = glasbey.create_palette(palette_size=4, colorblind_safe=True)

    palette='colorblind'

    sns.set_theme(style="whitegrid", font_scale=1.7)
    plt.figure(figsize=(10,9))
    for i, obj in enumerate(data):
        rewards = obj['rewards']
        plot_violin_sns(rewards, labels[i], "", "Average Reward over one Game")
    export_plot(f"{folder}/rewards.pdf")
    export_plot_png(f"{folder}/rewards.png")


    sns.set_theme(style="whitegrid", font_scale=1.7)
    plt.figure(figsize=(10,9))
    for i, obj in enumerate(data):
        lengths = obj['lengths']
        plot_violin_sns(lengths, labels[i], "", "Length of Game Until Win/Draw")
    export_plot(f"{folder}/lengths.pdf")
    export_plot_png(f"{folder}/lengths.png")


def load_pkl_files(files):
    data_arr = []
    for pkl_file in files:
        data_arr.append(load_statistics(pkl_file))
    return data_arr


def export_plot(filename):
    plt.tight_layout()
    plt.savefig(filename, format='pdf', dpi=300, bbox_inches='tight')
    print(f"Plot successfully saved as {filename}")

def export_plot_png(filename):
    plt.tight_layout()
    plt.savefig(filename, format='png', dpi=300, bbox_inches='tight')
    print(f"Plot successfully saved as {filename}")

