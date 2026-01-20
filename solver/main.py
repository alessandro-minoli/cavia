import os
import pickle

DATASET_NETWORK_DIR = "../dataset/network"
DATASET_APP_DIR = "../dataset/app"
EXPERIMENTS_DIR = "../experiments"

if not os.path.exists(EXPERIMENTS_DIR):
    os.makedirs(EXPERIMENTS_DIR)

from exceptions import InfeasibleInstance
from dataset_reader import instances_iterator
from instance import create_instance
from compact_model import build_compact_model, optimize_compact_model, optimize_compact_model_restricted
from utils import read_compact_model_logfile_custom
from cg_with_rounding import run_cg_with_rounding


NETWORK_SIZES = [30,40,50,60] # 70
NUMBER_OF_TOPOLOGIES_PER_SIZE = 3
NUMBER_OF_RP_FILES_PER_TOPOLOGY = 1
MIN_NAPPS = 5
MAX_NAPPS = 40
INSTANCES_PER_NAPPS = 3

for ins in instances_iterator(
    NETWORK_SIZES, NUMBER_OF_TOPOLOGIES_PER_SIZE, NUMBER_OF_RP_FILES_PER_TOPOLOGY,
    MIN_NAPPS, MAX_NAPPS, INSTANCES_PER_NAPPS
):
    
    experiment_dir = os.path.join(EXPERIMENTS_DIR, ins['experiment_id'])

    assert os.path.exists(experiment_dir)
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)
    
    # print(ins['experiment_id'])
    
    # --- compact_model -------------------------------------------------------

    compact_model_logfile_grb = os.path.join(experiment_dir, "compact_model_gurobi.log")
    compact_model_logfile_custom = os.path.join(experiment_dir, "compact_model_custom.log")
    assert os.path.exists(compact_model_logfile_grb) and os.path.exists(compact_model_logfile_custom)

    if not (os.path.exists(compact_model_logfile_grb) and os.path.exists(compact_model_logfile_custom)):
        print("... create_instance")
        try:
            instance = create_instance(
                DATASET_NETWORK_DIR, DATASET_APP_DIR, ins
            )
        except InfeasibleInstance:
            print("INFEASIBLE on building")
            continue

        print("... build_compact_model")
        model, x = build_compact_model(instance)
    
        print("... optimize_compact_model")
        time_limit_seconds = 15*60
        output = optimize_compact_model(model, time_limit_seconds, compact_model_logfile_custom, compact_model_logfile_grb)
        print(output)

    # --- cg_with_rounding ----------------------------------------------------

    _, gurobi_data = read_compact_model_logfile_custom(compact_model_logfile_custom)
    needs_cg_with_rounding = gurobi_data["Status"] in (">15m_SOL-FOUND", ">15m_NO-SOL-FOUND")

    if needs_cg_with_rounding:

        print(ins['experiment_id'])

        print("... create_instance")
        try:
            instance = create_instance(
                DATASET_NETWORK_DIR, DATASET_APP_DIR, ins
            )
        except InfeasibleInstance:
            print("INFEASIBLE on building")
            continue

        print("... build_compact_model")
        model, x = build_compact_model(instance)

        
        # print("... run_column_generation_with_rounding")
        for exact_pricing_rounds in [0,1]:

            cg_with_rounding_logfile = os.path.join(
                experiment_dir, 
                f"cg_with_rounding_result_exact-pricing-rounds_{exact_pricing_rounds}.pkl"
            )

            assert os.path.exists(cg_with_rounding_logfile)
            if os.path.exists(cg_with_rounding_logfile):
                with open(cg_with_rounding_logfile, 'rb') as f:
                    result = pickle.load(f)
            else:
                result = run_cg_with_rounding(
                    instance, 
                    {
                        "stop_gap" : 1,
                        "exact_pricing_rounds" : exact_pricing_rounds,
                        "max_fixed_non-mandatory_perc" : 0.7
                    }
                )
                with open(cg_with_rounding_logfile, 'wb') as f:
                    pickle.dump(result, f)

            # subMIP
            
            if result['root_node']['status'] in [("SUCCESS","NO-EXACT-PP_NOCOLFOUND_DUMMY-1"),("SUCCESS","GAP-1_DUMMY-1")]:
                assert result['rounding']['perc_non-mandatory_final_fixings'] is not None
                (
                    n_microservices_total, n_mandatory_fixings,
                    history_fixing_type, history_fixing_status, history_fixing_time,
                    n_nonmandatory_final_fixings, perc_nonmandatory_final_fixings,
                ) = (
                    result['rounding']['n_microservices_total'],
                    result['rounding']['n_mandatory_fixings'],
                    result['rounding']['history_fixing_type'], 
                    result['rounding']['history_fixing_status'], 
                    result['rounding']['history_fixing_time'],
                    result['rounding']['n_non-mandatory_final_fixings'], 
                    result['rounding']['perc_non-mandatory_final_fixings']
                )
                
                for FIXING_PERC in [0.3,0.4]:
                    fixing_is_possible = False
                    submip_fixings = []
                    time_fixings = None
                    for i in range(n_mandatory_fixings):
                        assert history_fixing_status[i] == ("SUCCESS","MANDATORY")
                        submip_fixings.append(history_fixing_type[i])
                    
                    cnt = 0
                    for i in range(n_mandatory_fixings, len(history_fixing_status)):
                        assert history_fixing_status[i] != ("SUCCESS","MANDATORY")
                        if history_fixing_status[i][0] != "SUCCESS":
                            assert i == len(history_fixing_status)-1
                            break
                        cnt += 1
                        submip_fixings.append(history_fixing_type[i])
                        cur_perc = cnt / (n_microservices_total-n_mandatory_fixings)
                        if cur_perc >= FIXING_PERC:
                            fixing_is_possible = True
                            time_fixings = history_fixing_time[i]
                            break
                    
                    fixing_perc_as_str = str(round(FIXING_PERC * 100))
                    submip_model_logfile_grb = os.path.join(experiment_dir, f"submip_{fixing_perc_as_str}_epr{exact_pricing_rounds}_model_gurobi.log")
                    submip_model_logfile_custom = os.path.join(experiment_dir, f"submip_{fixing_perc_as_str}_epr{exact_pricing_rounds}_model_custom.log")
                    if os.path.exists(submip_model_logfile_grb):
                        assert os.path.exists(submip_model_logfile_custom)
                        continue
                    assert not (os.path.exists(submip_model_logfile_grb) and os.path.exists(submip_model_logfile_custom))
                    if fixing_is_possible:
                        assert time_fixings is not None
                        time_limit_for_submip = 15*60 - (result['root_node']['time'] + time_fixings)
                        if time_limit_for_submip <= 0:
                            print("FAIL no_time_left_for_submip")
                            with open(submip_model_logfile_custom, 'w') as f:
                                f.write("FAIL no_time_left_for_submip")
                            with open(submip_model_logfile_grb, 'w') as f:
                                f.write("FAIL no_time_left_for_submip")
                        else:
                            print(f"SUCCESS starting a submip with time limit {time_limit_for_submip}")
                            output = optimize_compact_model_restricted(model, time_limit_for_submip, submip_model_logfile_custom, submip_model_logfile_grb, submip_fixings, instance, x)
                            print(output)
                    else:
                        print("FAIL cannot_reach_target_fixing_percentage")
                        with open(submip_model_logfile_custom, 'w') as f:
                            f.write("FAIL cannot_reach_target_fixing_percentage")
                        with open(submip_model_logfile_grb, 'w') as f:
                            f.write("FAIL cannot_reach_target_fixing_percentage")
    

# se le esecuzioni con pricing esatti ci mettono troppo,
# si puo' capire di migliorare il local search.
# per esempio partendo dalla colonna con costo ridotto piu negativo secondo i duali correnti