import os
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch

from utils import read_compact_model_logfile_custom
from dataset_reader import instances_iterator

EXPERIMENTS_DIR = "../experiments"

assert os.path.exists(EXPERIMENTS_DIR)

NETWORK_SIZES = [30,40,50,60] # 70
NUMBER_OF_TOPOLOGIES_PER_SIZE = 3
NUMBER_OF_RP_FILES_PER_TOPOLOGY = 1
MIN_NAPPS = 5
MAX_NAPPS = 40
INSTANCES_PER_NAPPS = 3

COMPACT_MODEL_DATA = dict()

### load COMPACT_MODEL_DATA 

for ins in instances_iterator(
    NETWORK_SIZES, NUMBER_OF_TOPOLOGIES_PER_SIZE, NUMBER_OF_RP_FILES_PER_TOPOLOGY,
    MIN_NAPPS, MAX_NAPPS, INSTANCES_PER_NAPPS
):

    key = (ins["network_size"], ins["napps"])

    if key not in COMPACT_MODEL_DATA:
        COMPACT_MODEL_DATA[key] = {
            "OPTIMAL" : 0,
            ">15m_SOL-FOUND" : 0,
            ">15m_NO-SOL-FOUND" : 0,
            "INFEASIBLE" : 0,
        }

    experiment_dir = os.path.join(EXPERIMENTS_DIR, ins['experiment_id'])
    # assert os.path.exists(experiment_dir)
    
    if os.path.exists(experiment_dir):  

        compact_model_logfile_custom = os.path.join(experiment_dir, "compact_model_custom.log")
        assert os.path.exists(compact_model_logfile_custom)

        _, gurobi_data = read_compact_model_logfile_custom(compact_model_logfile_custom)
        assert gurobi_data["Status"] in COMPACT_MODEL_DATA[key]
        COMPACT_MODEL_DATA[key][gurobi_data["Status"]] += 1

for k,v in COMPACT_MODEL_DATA.items():
    print(k,v)

### print COMPACT_MODEL_DATA


# Extract categories dynamically
network_sizes = sorted({ns for (ns, na) in COMPACT_MODEL_DATA.keys()})
napps_values  = sorted({na for (ns, na) in COMPACT_MODEL_DATA.keys()})

labels = ["OPTIMAL", ">15m_SOL-FOUND", ">15m_NO-SOL-FOUND", "INFEASIBLE"]

# Color mapping
color_map = {
    "OPTIMAL": "green",
    ">15m_SOL-FOUND": "lightgreen",
    ">15m_NO-SOL-FOUND": "lightcoral",
    "INFEASIBLE": "red"
}

# Create one subplot per network size
fig, axes = plt.subplots(
    nrows=len(network_sizes),
    ncols=1,
    figsize=(1.3 * len(napps_values), 4 * len(network_sizes)),
    sharex=True
)

# If only one network size, wrap in list
if len(network_sizes) == 1:
    axes = [axes]

for ax, ns in zip(axes, network_sizes):

    # Draw the 4 rows for this network size
    for r, label in enumerate(labels):
        for j, na in enumerate(napps_values):
            value = COMPACT_MODEL_DATA.get((ns, na), {}).get(label, None)

            # Determine color
            if value is None or value == 0:
                facecolor = "white"
            else:
                facecolor = color_map[label]

            rect = Rectangle((j, r), 1, 1,
                                facecolor=facecolor,
                                edgecolor="black",
                                linewidth=1.5)
            ax.add_patch(rect)

            if value not in (None, 0):
                ax.text(j + 0.5, r + 0.5, str(value),
                        ha="center", va="center",
                        fontsize=24, color="black")

    # Configure subplot
    ax.set_xlim(0, len(napps_values))
    ax.set_ylim(0, 4)
    ax.invert_yaxis()

    # Remove y-axis labels and ticks
    ax.set_yticks([])
    ax.set_yticklabels([])
    ax.set_ylabel("")

    # Title = network size
    ax.set_title(f"network size = {ns}", fontsize=28, pad=10)

# X-axis labels only on the bottom subplot
axes[-1].set_xticks([i + 0.5 for i in range(len(napps_values))])
axes[-1].set_xticklabels(napps_values, fontsize=24)
axes[-1].set_xlabel("#apps", fontsize=28)

# Legend (global)
legend_handles = [
    Patch(facecolor="green", edgecolor="black", label="OPTIMAL"),
    Patch(facecolor="lightgreen", edgecolor="black", label=">15m SOL-FOUND"),
    Patch(facecolor="lightcoral", edgecolor="black", label=">15m NO-SOL-FOUND"),
    Patch(facecolor="red", edgecolor="black", label="INFEASIBLE")
]
fig.legend(handles=legend_handles, loc="upper right", fontsize=24)

plt.tight_layout()
fig.savefig("plots/compact_model.pdf", format="pdf", dpi=300, bbox_inches="tight", transparent=True)
