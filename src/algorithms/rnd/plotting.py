import pickle
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

env_name = "Hockey-One-v0"
suffixes = ["wl_weak_only2", "wl_weak", "wl_weak_every2", "wl_strong_only2", "wl_strong", "wl_strong_every2"]
labels = ["Only first two (weak enemy)", "Constant updates (weak enemy)", "Every second episode (weak enemy)", "Only first two (strong enemy)", "Constant updates (strong enemy)", "Every second episode (strong enemy)"]  # Legend labels
# suffixes = ["500", "1000", "2000", "3000", "4000", "5000", "6000"]
# labels = ["500", "1000", "2000", "3000", "4000", "5000", "6000"]

smooth_window = 5000
all_stats = []
for suffix in suffixes:
    stats_path = f"models/rnd/Tests/rnd_{env_name}_{suffix}.pkl"
    with open(stats_path, "rb") as f:
        all_stats.append(pickle.load(f))

plt.figure(figsize=(10, 6))
sns.set_theme(style="whitegrid")

for stats, label in zip(all_stats, labels):
    extrinsic = stats['extrinsic']

    episodes = np.arange(1, len(extrinsic) + 1)
    if smooth_window > 1:
        smoothed = np.convolve(
            extrinsic, 
            np.ones(smooth_window)/smooth_window, 
            mode='valid'
        )
        smoothed_episodes = episodes[smooth_window-1:]
        plt.plot(smoothed_episodes, smoothed, label=label)
    else:
        plt.plot(episodes, extrinsic, label=label)

plt.xlabel("Episodes")
plt.ylabel("Extrinsic Reward")
#plt.title(f"Update frequency Comparison (Env: {env_name})")
plt.title(f"Self-play settings Comparison (Env: {env_name})")
filename = f"models/rnd/Tests2/rnd_{env_name}_plot2.pdf"
#filename = f"models/rnd/Tests2/rnd_{env_name}_plot.pdf"
plt.legend()
plt.grid(True)
plt.savefig(filename, format='pdf', dpi=300, bbox_inches='tight')
plt.show()