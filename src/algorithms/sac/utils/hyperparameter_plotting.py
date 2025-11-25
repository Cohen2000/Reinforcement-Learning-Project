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
    pattern = r"alpha_([^_]+)_gamma_([^_]+)\.json"
    match = re.match(pattern, basename)
    if match:
        alpha = match.group(1)
        gamma = match.group(2)
        return alpha, gamma
    return None, None

def main():
    parser = argparse.ArgumentParser(description="Plotting experiments from JSON-Files")
    parser.add_argument("--folder", required=True, help="Folder containing JSON files")
    args = parser.parse_args()

    folder_path = os.path.join(os.getcwd(), args.folder)
    json_files = glob.glob(os.path.join(folder_path, "*.json"))

    if not json_files:
        print(f"No JSON-files found in folder: {folder_path}")
        return

    alpha_order = {"auto": 0, "0": 1, "0.2": 2}

    data_list = []
    for file in json_files:
        basename = os.path.basename(file)
        if basename.startswith("no") or basename.startswith("pp"):
            continue

        try:
            with open(file, "r") as f:
                data = json.load(f)
            win_rate_points = data.get("win_rate_points", [])
            if not win_rate_points:
                continue

            x_values = [point[0] for point in win_rate_points]
            y_values = [point[1] for point in win_rate_points]

            alpha, gamma = parse_filename(file)
            if alpha is None or gamma is None:
                label = basename
            else:
                label = f"α: {alpha}, γ: {gamma}"
            data_list.append((alpha, gamma, label, x_values, y_values))
        except Exception as e:
            print(f"Error while loading {file}: {e}")

    def sort_key(item):
        alpha, gamma, _, _, _ = item
        order_alpha = alpha_order.get(alpha, 99)
        try:
            gamma_val = float(gamma)
        except:
            gamma_val = 0
        return (order_alpha, gamma_val)

    data_list.sort(key=sort_key)

    sns.set(style="whitegrid")
    plt.figure(figsize=(8, 6))

    for _, _, label, x_vals, y_vals in data_list:
        sns.lineplot(x=x_vals, y=y_vals, label=label, marker="o")

    plt.xlabel("Episodes")
    plt.ylabel("Win Rate (%)")
    plt.xlim(0, 15000)
    plt.xticks(np.arange(0, 15001, 1000))
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    plt.legend(title="Experiments", loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False)
    plt.subplots_adjust(bottom=0.25)
    plt.tight_layout()
    plt.savefig("hyperparam_plot.pdf", format='pdf', dpi=300, bbox_inches='tight')
    print("Plot successfully saved as hyperparam_plot.pdf")

if __name__ == "__main__":
    main()
