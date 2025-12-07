import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def parse_file(path):
    results = {}  # {system: {method: mine_val}}
    system = None
    with open(path, "r") as f:
        for line in f:
            sys_match = re.match(r'-+([A-Z_]+)-+', line)
            if sys_match:
                system = sys_match.group(1)
            row_match = re.match(r'(GOKU|Latent-ODE|LSTM)\s*\|\s*([\d\.]+) \(mine\)', line)
            if row_match and system:
                method = row_match.group(1)
                mine_val = float(row_match.group(2))
                if system not in results:
                    results[system] = {}
                results[system][method] = mine_val
    return results

def custom_bar_and_trend(data_dicts, labels, methods=["GOKU", "Latent-ODE", "LSTM"]):
    systems = ["DOUBLE_PENDULUM", "PENDULUM", "CVS"]
    system_titles = {'CVS': 'CVS', 'PENDULUM': 'Pendulum', 'DOUBLE_PENDULUM': 'Double Pendulum'}
    colors = ['dodgerblue', 'orangered']  # AdamW, Adam
    x = np.arange(len(methods))
    width = 0.35

    fig = plt.figure(figsize=(12, 9))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1.1])
    # Top row: Double Pendulum [0,0], Pendulum [0,1]
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    # Bottom row: CVS spans both columns
    ax2 = fig.add_subplot(gs[1, :])

    axes = [ax0, ax1, ax2]
    for idx, (system, ax) in enumerate(zip(systems, axes)):
        for i, data in enumerate(data_dicts):
            vals = [data.get(system, {}).get(m, np.nan) for m in methods]
            pos = x - width/2 + i*width
            bars = ax.bar(pos, vals, width, align='center', color=colors[i], alpha=0.6, label=labels[i])
            ax.plot(x, vals, marker='o', color=colors[i], linewidth=2, alpha=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels(methods)
        ax.set_title(system_titles[system])
        ax.set_ylabel('L1 Error (x1e3)')
        ax.set_ylim(bottom=0)
        ax.grid(True, linestyle=":", alpha=0.7)
        if idx == 0:
            # Only show legend once, for clarity
            ax.legend()
    plt.suptitle("Extrapolation L1 Error (mine): AdamW vs Adam\nDouble Pendulum & Pendulum (Top), CVS (Bottom)", fontsize=15)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

if __name__ == "__main__":
    adamw_results = parse_file("evaluate/adamw_output.txt")
    adam_results = parse_file("evaluate/adam_output.txt")
    custom_bar_and_trend([adamw_results, adam_results], labels=["AdamW", "Adam"])