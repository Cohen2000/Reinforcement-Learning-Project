import os
import json
import argparse
import glob
import re
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def parse_filename(filename):
    basename = os.path.basename(filename)
    is_no_shaping = False
    if basename.startswith("no_shaping_"):
        is_no_shaping = True
        basename = basename.replace("no_shaping_", "", 1)
    pattern = r"alpha_([^_]+)_gamma_([^_]+)\.json"
    match = re.match(pattern, basename)
    if match:
        alpha = match.group(1)
        gamma = match.group(2)
        return alpha, gamma, is_no_shaping
    return None, None, is_no_shaping

def main():
    parser = argparse.ArgumentParser(description="Plot Reward Shaping Comparison Experiments")
    parser.add_argument("--folder", required=True, help="Folder containing the JSON files")
    args = parser.parse_args()

    folder_path = os.path.abspath(args.folder)
    json_files = glob.glob(os.path.join(folder_path, "*.json"))
    if not json_files:
        print(f"No JSON files found in folder: {folder_path}")
        return

    allowed_combinations = {("auto", "0.9"), ("auto", "0.95"), ("0", "0.95")}
    alpha_order = {"auto": 0, "0": 1}

    color_mapping = {
        ("auto", "0.95"): "#ff7f0e",  # orange
        ("auto", "0.9"): "#1f77b4",   # blue
        ("0", "0.95"): "#9467bd",     # purple 
    }

    data_list = []
    for file in json_files:
        try:
            with open(file, "r") as f:
                data = json.load(f)
            win_rate_points = data.get("win_rate_points", [])
            if not win_rate_points:
                continue

            alpha, gamma, is_no_shaping = parse_filename(file)
            if alpha is None or gamma is None:
                continue

            if (alpha, gamma) not in allowed_combinations:
                continue

            x_values = [point[0] for point in win_rate_points]
            y_values = [point[1] for point in win_rate_points]
            shaping_status = "No Shaping" if is_no_shaping else "Shaping"
            label = f"{shaping_status}: α={alpha}, γ={gamma}"
            data_list.append((is_no_shaping, alpha, gamma, label, x_values, y_values))
        except Exception as e:
            print(f"Error loading {file}: {e}")

    data_no_shaping = [d for d in data_list if d[0]]
    data_shaping = [d for d in data_list if not d[0]]

    def sort_key(item):
        _, alpha, gamma, _, _, _ = item
        order_alpha = alpha_order.get(alpha, 99)
        try:
            gamma_val = float(gamma)
        except:
            gamma_val = 0
        return (order_alpha, gamma_val)

    data_no_shaping.sort(key=sort_key)
    data_shaping.sort(key=sort_key)

    sns.set(style="whitegrid")
    plt.figure(figsize=(8, 6))

    for is_no_shaping, alpha, gamma, label, x_vals, y_vals in data_shaping:
        color = color_mapping.get((alpha, gamma), None)
        sns.lineplot(x=x_vals, y=y_vals, label=label, marker="o", linestyle="solid", color=color)

    for is_no_shaping, alpha, gamma, label, x_vals, y_vals in data_no_shaping:
        color = color_mapping.get((alpha, gamma), None)
        sns.lineplot(x=x_vals, y=y_vals, label=label, marker="o", linestyle="dashed", color=color)

    plt.xlabel("Episodes")
    plt.ylabel("Win Rate (%)")
    plt.xlim(0, 15000)
    plt.xticks(np.arange(0, 15001, 1000))
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.legend(title="Experiments", loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False)
    plt.subplots_adjust(bottom=0.25)
    
    plt.tight_layout()
    plt.savefig("shaping_plot.pdf", format='pdf', dpi=300, bbox_inches='tight')
    print(f"Plot successfully saved as shaping_plot.pdf")

if __name__ == "__main__":
    main()
