from time import perf_counter

from exceptions import InfeasiblePricing
from cg_master import Master
from cg_pricer import Pricer
from cg_pricer_local_search import PricerLocalSearch

EPSILON = 0.001

def run_cg_with_rounding(ins, input):

    ### INPUT PARAMS
    
    STOP_GAP                    = input["stop_gap"]
    EXACT_PRICING_ROUNDS        = input["exact_pricing_rounds"]
    MAX_FIXED_NONMANDATORY_PERC = input["max_fixed_non-mandatory_perc"]

    ### OUTPUT

    RESULT = {
        "input" : input,
        "root_node" : {
            "status" : None,
            "time" : None, 
            "n_iterations" : None,
            # history of mp_objective_value, mp_time, pp_type (H/E), pp_time, best_dual_bound
        },
        "rounding" : {
            "n_microservices_total" : None,
            "n_mandatory_fixings" : None,
            "history_fixing_type" : None,
            "history_fixing_status" : None,
            "history_fixing_time" : None,
            "n_non-mandatory_final_fixings" : None,
            "perc_non-mandatory_final_fixings" : None,
        }
    }

    """
        RESULT["root_node"]["status"] :
            ("INFEASIBLE",  "PP_INFEASIBLE"                 )   : there is an app that cannot be mapped even individually
            ("FAIL",        "NO-EXACT-PP_NOCOLFOUND_DUMMY-0")   : only heuristic pricing is not enough to remove dummy    
            ("SUCCESS",     "NO-EXACT-PP_NOCOLFOUND_DUMMY-1")   : only heuristic pricing is     enough to remove dummy
            ("SUCCESS",     "GAP-1_DUMMY-1"                 )   : gap is ok and dummy are not used
            ("INFEASIBLE",  "NOCOLFOUND_GAP-1_DUMMY-0"      )   : gap is ok but dummy cannot be removed
            ("INFEASIBLE",  "NOCOLFOUND_GAP-0_DUMMY-1"      )   : expected to never happen
            ("INFEASIBLE",  "NOCOLFOUND_GAP-0_DUMMY-0"      )   : expected to never happen
            ("SUCCESS",     "INTEGER_SOL"                   )   : cg root node solution is already integer

        RESULT["rounding"]["history_fixing_status"] elements can be :

            ("SUCCESS",     "MANDATORY"                     )   
            ("SUCCESS",     "ZERO_REMOVED_COLUMNS"          )
            ("FAIL",        "NO-EXACT-PP_NOCOLFOUND_DUMMY-0")   : * with the current fixings, only heuristic pricing is not enough to remove dummy
            ("SUCCESS",     "NO-EXACT-PP_NOCOLFOUND_DUMMY-1")   :   with the current fixings, only heuristic pricing is     enough to remove dummy
            ("INFEASIBLE",  "PP_INFEASIBLE"                 )   : * with the current fixings, there is an app that cannot be mapped even individually
            ("SUCCESS",     "GAP-1_DUMMY-1"                 )   :   with the current fixings, gap is ok and dummy are not used
            ("INFEASIBLE",  "NOCOLFOUND_GAP-1_DUMMY-0"      )   : * with the current fixings, gap is ok but dummy cannot be removed
            ("INFEASIBLE",  "NOCOLFOUND_GAP-0_DUMMY-1"      )   : * expected to never happen
            ("INFEASIBLE",  "NOCOLFOUND_GAP-0_DUMMY-0"      )   : * expected to never happen
            ("SUCCESS",     "INTEGER_SOL"                   )   :   with the current fixings, cg solution is already integer

        when * , last fixing must be ignored
    """

    apps = ins.apps
    N = len(apps)

    MP = Master(ins)
    PPs = [Pricer(ins,n) for n in range(N)]
    PPs_local_search = [PricerLocalSearch(ins,n) for n in range(N)]

    # --- root_node -----------------------------------------------------------

    start_root_node = perf_counter()

    # adding columns corresponding to the optimal mapping of the individual apps

    for n in range(N):

        try:
            col, _ = PPs[n].optimize(λ_positive=(), μ_positive=(), η_n=0)
        except InfeasiblePricing:
            RESULT["root_node"]["time"] = perf_counter() - start_root_node
            RESULT["root_node"]["status"] = "INFEASIBLE", "PP_INFEASIBLE"
            return RESULT
        
        MP.add_column(n, col)

    iterations_count = 0
    exact_iterations_left = 0
    best_dual_bound = 0

    while True:

        iterations_count += 1
        # print()
        # print(f"iter {iterations_count}")
        
        λ_positive, μ_positive, μ, η, mp_objective_value = MP.optimize() # non deve essere infeasible perchè ho le dummy
        # print(f"master {mp_objective_value}")

        cols_to_add = [None] * N
        negative_rc_col_found = False
        sum_reduced_costs = 0
        
        if exact_iterations_left == 0:

            # heuristic pricing
            
            for n in range(N):
                
                starting_col = MP.get_most_recent_mapping_of_app(n)
                new_col, reduced_cost = PPs_local_search[n].optimize(λ_positive, μ, η[n], starting_col) # non deve essere infeasible
                # print(f"{n} PP_ls : rc {reduced_cost} column {new_col.original_x if new_col is not None else None}")
                
                if reduced_cost < -EPSILON:
                    negative_rc_col_found = True
                    # assert cols_to_add[n] is None
                    cols_to_add[n] = new_col

            if negative_rc_col_found: 
                # assert cols_to_add != [None] * N

                for n in range(N):
                    if cols_to_add[n] is not None:
                        MP.add_column(n, cols_to_add[n])

            else:
                # assert cols_to_add == [None] * N

                if EXACT_PRICING_ROUNDS == 0:
                    if MP.still_using_dummy():
                        RESULT["root_node"]["time"] = perf_counter() - start_root_node
                        RESULT["root_node"]["status"] = "FAIL", "NO-EXACT-PP_NOCOLFOUND_DUMMY-0"
                        RESULT["root_node"]["n_iterations"] = iterations_count
                        return RESULT
                    else:
                        RESULT["root_node"]["time"] = perf_counter() - start_root_node
                        RESULT["root_node"]["status"] = "SUCCESS", "NO-EXACT-PP_NOCOLFOUND_DUMMY-1"
                        RESULT["root_node"]["n_iterations"] = iterations_count
                        break
                else: 
                    exact_iterations_left = EXACT_PRICING_ROUNDS

        else:

            # exact pricing

            for n in range(N):
                
                new_col, reduced_cost = PPs[n].optimize(λ_positive, μ_positive, η[n]) # non deve essere infeasible
                # print(f"{n} PP_ex : rc {reduced_cost} column {new_col.original_x if new_col is not None else None}")
                
                if reduced_cost < -EPSILON:
                    sum_reduced_costs += reduced_cost
                    negative_rc_col_found = True
                    # assert cols_to_add[n] is None
                    cols_to_add[n] = new_col

            lagrangean_dual_bound = mp_objective_value + sum_reduced_costs
            best_dual_bound = max(best_dual_bound, lagrangean_dual_bound)
            gap = mp_objective_value-best_dual_bound
            perc_gap = 100 * gap / mp_objective_value

            # print(f"perc_gap = {perc_gap}")

            gap_ok = perc_gap <= STOP_GAP
            dummy_ok = not MP.still_using_dummy()

            if gap_ok and dummy_ok:
                RESULT["root_node"]["time"] = perf_counter() - start_root_node
                RESULT["root_node"]["status"] = "SUCCESS", "GAP-1_DUMMY-1"
                RESULT["root_node"]["n_iterations"] = iterations_count
                break
            
            if negative_rc_col_found: 
                # assert cols_to_add != [None] * N

                for n in range(N):
                    if cols_to_add[n] is not None:
                        MP.add_column(n, cols_to_add[n])

            else:
                # assert cols_to_add == [None] * N
                RESULT["root_node"]["time"] = perf_counter() - start_root_node
                RESULT["root_node"]["status"] = "INFEASIBLE", f"NOCOLFOUND_GAP-{int(gap_ok)}_DUMMY-{int(dummy_ok)}"
                RESULT["root_node"]["n_iterations"] = iterations_count
                return RESULT
            
            exact_iterations_left -= 1

    if MP.solution_is_integer():
        RESULT["root_node"]["time"] = perf_counter() - start_root_node
        RESULT["root_node"]["status"] = "SUCCESS", "INTEGER_SOL"
        RESULT["root_node"]["n_iterations"] = iterations_count
        return RESULT

    # --- rounding ------------------------------------------------------------

    start_rounding = perf_counter()

    n_microservices_total = sum(app.T for app in apps)
    history_fixing_time = []
    history_fixing_type = []
    history_fixing_status = []
    n_fixed_microservices = 0
    n_mandatory_fixings = 0
    still_to_fix = [set(range(apps[n].T)) for n in range(N)]

    # do the mandatory fixings, that are the ones for which the microservices has only 1 candidate node
    for n in range(N):
        for u in range(apps[n].T):
            if len(apps[n].b_microservices_one[u]) == 1:
                history_fixing_time.append(perf_counter()-start_rounding)
                history_fixing_type.append((n,u,apps[n].b_microservices_one[u][0]))
                history_fixing_status.append(("SUCCESS","MANDATORY"))
                n_fixed_microservices += 1
                n_mandatory_fixings += 1
                still_to_fix[n].remove(u)

    RESULT["rounding"]["n_microservices_total"] = n_microservices_total
    RESULT["rounding"]["n_mandatory_fixings"] = n_mandatory_fixings

    fixings = [[] for _ in range(N)] # questa struttura si puo' eliminare poi

    max_to_fix = n_mandatory_fixings + MAX_FIXED_NONMANDATORY_PERC * (n_microservices_total-n_mandatory_fixings)

    while n_fixed_microservices < max_to_fix:

        n_most, u_most, i_most, _ = MP.get_most_selected_fixing(still_to_fix)

        # print(f"app {n_most} : fix {u_most} on {i_most}")

        removed_columns_count = MP.remove_columns_of_n_in_which_u_is_not_on_i(n_most, u_most, i_most)
        # print("removed_columns_count ", removed_columns_count)
        
        PPs[n_most].fix_u_on_i(u_most,i_most)
        PPs_local_search[n_most].remove_from_movable_microservices(u_most)

        history_fixing_type.append((n_most,u_most,i_most))
        n_fixed_microservices += 1
        still_to_fix[n_most].remove(u_most)
        
        fixings[n_most].append((u_most,i_most))

        if removed_columns_count == 0:
            history_fixing_time.append(perf_counter()-start_rounding)
            history_fixing_status.append(("SUCCESS","ZERO_REMOVED_COLUMNS"))
            # assert len(history_fixing_type) == len(history_fixing_time) == len(history_fixing_status)
            continue

        # re-optimize with column generation

        status = None
        iterations_count = 0
        exact_iterations_left = 0
        best_dual_bound = 0

        stop_fixing = False

        while True:

            iterations_count += 1
            # print()
            # print(f"iter {iterations_count}")
            
            λ_positive, μ_positive, μ, η, mp_objective_value = MP.optimize()  # non deve essere infeasible perchè ho le dummy
            # print(f"master {mp_objective_value}")

            cols_to_add = [None] * N
            negative_rc_col_found = False
            sum_reduced_costs = 0
            
            if exact_iterations_left == 0:

                # heuristic pricing
                
                for n in range(N):
                    
                    starting_col = MP.get_most_recent_mapping_of_app(n)
                    new_col, reduced_cost = PPs_local_search[n].optimize(λ_positive, μ, η[n], starting_col, fixings[n])
                    # print(f"{n} PP_ls : rc {reduced_cost} column {new_col.original_x if new_col is not None else None}")
                    
                    if reduced_cost < -EPSILON:
                        negative_rc_col_found = True
                        # assert cols_to_add[n] is None
                        cols_to_add[n] = new_col

                if negative_rc_col_found: 
                    # assert cols_to_add != [None] * N

                    for n in range(N):
                        if cols_to_add[n] is not None:
                            MP.add_column(n, cols_to_add[n])

                else:
                    # assert cols_to_add == [None] * N

                    if EXACT_PRICING_ROUNDS == 0:
                        if MP.still_using_dummy():
                            status = "FAIL", "NO-EXACT-PP_NOCOLFOUND_DUMMY-0"
                            stop_fixing = True
                            break
                        else:
                            status = "SUCCESS", "NO-EXACT-PP_NOCOLFOUND_DUMMY-1"
                            break
                    else: 
                        exact_iterations_left = EXACT_PRICING_ROUNDS

            else:

                # exact pricing

                for n in range(N):
                    
                    try:
                        new_col, reduced_cost = PPs[n].optimize(λ_positive, μ_positive, η[n])
                        # print(f"{n} PP_ex : rc {reduced_cost} column {new_col.original_x if new_col is not None else None}")
                    except InfeasiblePricing:
                        status = "INFEASIBLE", "PP_INFEASIBLE"
                        stop_fixing = True
                        break
            
                    if reduced_cost < -EPSILON:
                        sum_reduced_costs += reduced_cost
                        negative_rc_col_found = True
                        # assert cols_to_add[n] is None
                        cols_to_add[n] = new_col
                
                if stop_fixing: break

                lagrangean_dual_bound = mp_objective_value + sum_reduced_costs
                best_dual_bound = max(best_dual_bound, lagrangean_dual_bound)
                gap = mp_objective_value-best_dual_bound
                perc_gap = 100 * gap / mp_objective_value

                gap_ok = perc_gap <= STOP_GAP
                dummy_ok = not MP.still_using_dummy()

                if gap_ok and dummy_ok:
                    status = "SUCCESS", "GAP-1_DUMMY-1"
                    break
                
                if negative_rc_col_found: 
                    # assert cols_to_add != [None] * N

                    for n in range(N):
                        if cols_to_add[n] is not None:
                            MP.add_column(n, cols_to_add[n])

                else:
                    # assert cols_to_add == [None] * N
                    status = "INFEASIBLE", f"NOCOLFOUND_GAP-{int(gap_ok)}_DUMMY-{int(dummy_ok)}"
                    stop_fixing = True
                    break
                
                exact_iterations_left -= 1

        # assert status is not None 

        if status[0] == "SUCCESS" and MP.solution_is_integer():
            status = "SUCCESS", "INTEGER_SOL"
            stop_fixing = True
        
        history_fixing_time.append(perf_counter()-start_rounding)
        history_fixing_status.append(status)
        # assert len(history_fixing_type) == len(history_fixing_time) == len(history_fixing_status)
        
        if stop_fixing:
            break
    
    RESULT["rounding"]["history_fixing_type"] = history_fixing_type
    RESULT["rounding"]["history_fixing_status"] = history_fixing_status
    RESULT["rounding"]["history_fixing_time"] = history_fixing_time
    RESULT["rounding"]["n_non-mandatory_final_fixings"] = (n_fixed_microservices - n_mandatory_fixings) 
    RESULT["rounding"]["perc_non-mandatory_final_fixings"] = (n_fixed_microservices - n_mandatory_fixings) / (n_microservices_total - n_mandatory_fixings)
    
    """
    print("fixed", sum(n_fixed_microservices))
    print("remaining", n_microservices_total-sum(n_fixed_microservices))
    print("... building instance")
    instance = Instance.build(
        network_filename, 
        network_rp_filename, 
        app_filename=apps_merged_filename, 
        app_rp_filename=apps_merged_rp_filename
    )
    end_R = perf_counter()
    elapsed_minutes = (end_R-start_CG) / 60
    print(f"... solving restricted compact model {end_R-start_CG:.3f} with time limit {15-elapsed_minutes}")

    res = ModelSolver.optimize_model_restricted(instance, history_fixing_type, T_n, start_CG, 15-elapsed_minutes)
    end_R = perf_counter()
    print(f"{end_R-start_CG:.3f} result {res}")
    """

    return RESULT
