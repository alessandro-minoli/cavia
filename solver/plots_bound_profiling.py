import os
import matplotlib.pyplot as plt
# from matplotlib.patches import Rectangle, Patch

from utils import read_compact_model_logfile_custom, read_submip_model_logfile_custom
from dataset_reader import instances_iterator

EXPERIMENTS_DIR = "../experiments"

assert os.path.exists(EXPERIMENTS_DIR)

NETWORK_SIZES = [30,40,50,60] # 70
NUMBER_OF_TOPOLOGIES_PER_SIZE = 3
NUMBER_OF_RP_FILES_PER_TOPOLOGY = 1
MIN_NAPPS = 5
MAX_NAPPS = 40
INSTANCES_PER_NAPPS = 3

submip_types = ["30_epr0", "30_epr1", "40_epr0", "40_epr1"]

COUNTER = {
    ">15m_NO-SOL-FOUND and NO-SOL-FOUND by any submip" : 0,
    ">15m_NO-SOL-FOUND and SOL-FOUND by some submip" : 0,
    ">15m_SOL-FOUND and NO-SOL-FOUND by any submip" : 0,
    ">15m_SOL-FOUND and SOL-FOUND by some submip" : 0
}

COUNTER_NOSOL = { t : 0 for t in submip_types }
COUNTER_SOL = { t : 0 for t in submip_types }

### load data 

for ins in instances_iterator(
    NETWORK_SIZES, NUMBER_OF_TOPOLOGIES_PER_SIZE, NUMBER_OF_RP_FILES_PER_TOPOLOGY,
    MIN_NAPPS, MAX_NAPPS, INSTANCES_PER_NAPPS
):
    
    experiment_dir = os.path.join(EXPERIMENTS_DIR, ins['experiment_id'])
    assert os.path.exists(experiment_dir)
    
    # if os.path.exists(experiment_dir):  

    compact_model_logfile_custom = os.path.join(experiment_dir, "compact_model_custom.log")
    assert os.path.exists(compact_model_logfile_custom)

    TO_PLOT = []

    bound_profiling_data, gurobi_data = read_compact_model_logfile_custom(compact_model_logfile_custom)

    if gurobi_data["Status"] in [">15m_SOL-FOUND", ">15m_NO-SOL-FOUND"]:
        print("... plotting bound_profiling of", ins['experiment_id'])

        if gurobi_data["Status"] == ">15m_NO-SOL-FOUND":
            assert len(bound_profiling_data) == 0
        TO_PLOT.append(
            (bound_profiling_data, f"grb - {gurobi_data["Status"]}", 'grb')
        )

        for t in submip_types:
            submip_model_logfile_custom = os.path.join(experiment_dir, f"submip_{t}_model_custom.log")
            if not os.path.exists(submip_model_logfile_custom):
                continue

            submip_bound_profiling_data, submip_gurobi_data = read_submip_model_logfile_custom(submip_model_logfile_custom)
            if submip_gurobi_data["Status"] in (">15m_NO-SOL-FOUND", "INFEASIBLE", "FAIL"):
                assert len(submip_bound_profiling_data) == 0
            else:
                assert len(submip_bound_profiling_data) > 0
            TO_PLOT.append(
                (submip_bound_profiling_data, f"{t} - {submip_gurobi_data["Status"]}", t)
            )

        if gurobi_data["Status"] == ">15m_NO-SOL-FOUND":
            if all(len(x) == 0 for x,_,_ in TO_PLOT[1:]): 
                COUNTER[">15m_NO-SOL-FOUND and NO-SOL-FOUND by any submip"] += 1
                continue
            else:
                COUNTER[">15m_NO-SOL-FOUND and SOL-FOUND by some submip"] += 1
                for x,_,t in TO_PLOT[1:]:
                    if len(x) > 0:
                        COUNTER_NOSOL[t] += 1

        else:
            if all(len(x) == 0 for x,_,_ in TO_PLOT[1:]): 
                COUNTER[">15m_SOL-FOUND and NO-SOL-FOUND by any submip"] += 1
            else:
                COUNTER[">15m_SOL-FOUND and SOL-FOUND by some submip"] += 1
                for x,_,t in TO_PLOT[1:]:
                    if len(x) > 0:
                        COUNTER_SOL[t] += 1

        plt.figure(figsize=(8, 5))
        for bound_profiling_data, label, _ in TO_PLOT:

            objectives = [x[0] for x in bound_profiling_data]
            times = [x[1] for x in bound_profiling_data]

            plt.step(times, objectives, where='post', linewidth=2, label=label)

        plt.xlim((0,905))
        plt.xlabel("time")
        plt.ylabel("z")
        plt.title(f"Bound Profiling - {ins['experiment_id']}")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"plots/bound_profiling/{ins['experiment_id']}.pdf", format="pdf", dpi=300, bbox_inches="tight", transparent=True)
        plt.close()

print()
print(f"# >15m_NO-SOL-FOUND and NO-SOL-FOUND by any submip : {COUNTER['>15m_NO-SOL-FOUND and NO-SOL-FOUND by any submip']}")
print(f"# >15m_SOL-FOUND    and NO-SOL-FOUND by any submip : {COUNTER['>15m_SOL-FOUND and NO-SOL-FOUND by any submip']}")
print(f"# >15m_NO-SOL-FOUND and SOL-FOUND by some submip : {COUNTER['>15m_NO-SOL-FOUND and SOL-FOUND by some submip']}")
for k,v in COUNTER_NOSOL.items():
    print(f"\t{k} : {v}")
print(f"# >15m_SOL-FOUND    and SOL-FOUND by some submip : {COUNTER['>15m_SOL-FOUND and SOL-FOUND by some submip']}")
for k,v in COUNTER_SOL.items():
    print(f"\t{k} : {v}")