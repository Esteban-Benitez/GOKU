import re
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import re

def parse_file(filename):
    data = {}
    current_block = None
    with open(filename, "r") as f:
        lines = f.readlines()
    for line in lines:
        # Block detection (more robust, matches even if extra dashes or whitespace)
        m_block = re.search(r'-+([A-Z_]+)-+', line)
        if m_block:
            block_name = m_block.group(1)
            current_block = block_name
            if current_block not in data:
                data[current_block] = {}
            continue

        # Match: [GOKU|Latent-ODE|LSTM] | float (mine)
        m_entry = re.match(r"\s*(GOKU|Latent-ODE|LSTM)\s*\|\s*([\d\.]+)\s*\(mine\)", line)
        if current_block and m_entry:
            model = m_entry.group(1)
            value = float(m_entry.group(2))
            data[current_block][model] = value

    return data
    
def plot_grouped_bar(ax, methods, adam_vals, adamw_vals, title, show_ylabel=False):
    x = range(len(methods))
    width = 0.35
    ax.bar([i - width/2 for i in x], adam_vals, width, label='Adam')
    ax.bar([i + width/2 for i in x], adamw_vals, width, label='AdamW')
    ax.set_xticks(list(x))
    ax.set_xticklabels(methods, rotation=45)
    ax.set_title(title)
    if show_ylabel:
        ax.set_ylabel('Extrap. L1 (x1e3) [pixels]')
    ax.legend()
    ax.grid(axis='y')

def main():
    adam_data = parse_file("evaluate/adam_output.txt")
    adamw_data = parse_file("evaluate/adamw_output.txt")

    methods = ["GOKU", "Latent-ODE", "LSTM"]
    block_titles = {
        "PENDULUM": "Pendulum",
        "DOUBLE_PENDULUM": "Double Pendulum",
        "CVS": "CVS"
    }

    # Layout: 2 cols x 2 rows, but bottom CVS plot spans both columns
    fig = plt.figure(figsize=(11, 8))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1,1.2])

    # Top row
    ax_pendulum = fig.add_subplot(gs[0, 0])
    ax_double = fig.add_subplot(gs[0, 1])
    plot_grouped_bar(
        ax_pendulum, methods, 
        [adam_data["PENDULUM"].get(m, None) for m in methods],
        [adamw_data["PENDULUM"].get(m, None) for m in methods],
        block_titles["PENDULUM"], 
        show_ylabel=True
    )
    plot_grouped_bar(
        ax_double, methods, 
        [adam_data["DOUBLE_PENDULUM"].get(m, None) for m in methods],
        [adamw_data["DOUBLE_PENDULUM"].get(m, None) for m in methods],
        block_titles["DOUBLE_PENDULUM"]
    )

    # Bottom row: CVS spans both columns
    ax_cvs = fig.add_subplot(gs[1, :])
    plot_grouped_bar(
        ax_cvs, methods, 
        [adam_data["CVS"].get(m, None) for m in methods],
        [adamw_data["CVS"].get(m, None) for m in methods],
        block_titles["CVS"],
        show_ylabel=True
    )

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()