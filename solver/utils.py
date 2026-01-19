def check_gurobi_data(gurobi_data):
    if gurobi_data['Status'] == 'OPTIMAL':
        assert gurobi_data['MIPGap'] == '0.0'
    elif gurobi_data['Status'] == 'INFEASIBLE':
        assert gurobi_data['ObjVal'] == 'None'
        assert gurobi_data['MIPGap'] == 'inf'
    else: 
        assert gurobi_data['Status'] == 'TIME_LIMIT'
        assert (
            (gurobi_data['ObjVal'] == 'inf' and gurobi_data['MIPGap'] == 'inf') or
            (float(gurobi_data['ObjVal']) > 0 and float(gurobi_data['MIPGap']) > 0)
        )
        if gurobi_data['ObjVal'] == 'inf' and gurobi_data['MIPGap'] == 'inf':
            gurobi_data['Status'] = ">15m_NO-SOL-FOUND"
        else:
            gurobi_data['Status'] = ">15m_SOL-FOUND"

def read_compact_model_logfile_custom(path):
    bound_profiling_data = []

    gurobi_data = dict()
    with open(path, 'r') as f:
        lines = list(map(lambda x: x.strip(), f.readlines()))
        i = lines.index("GUROBI")
        assert lines[i-1].startswith("time ")
        for l in lines[:i-1]:
            _, sol, _, time = l.split()
            sol, time = float(sol), float(time)
            assert sol % 1 < 0.001 or sol % 1 > 1-0.001
            sol = round(sol)
            if len(bound_profiling_data) == 0 or sol < bound_profiling_data[-1][0]:
                bound_profiling_data.append((sol,time))
        for l in lines[i+1:]:
            k,v = l.split()
            gurobi_data[k] = v

    check_gurobi_data(gurobi_data)

    return bound_profiling_data, gurobi_data


def read_submip_model_logfile_custom(path):
    bound_profiling_data = []

    gurobi_data = dict()
    with open(path, 'r') as f:
        lines = list(map(lambda x: x.strip(), f.readlines()))
        if len(lines) == 1:
            assert lines[0] in (
                'FAIL cannot_reach_target_fixing_percentage',
                'FAIL no_time_left_for_submip'
            )
            gurobi_data['Status'] = "FAIL"
            return bound_profiling_data, gurobi_data

        j = lines.index("FIXINGS_END")
        i = lines.index("GUROBI")
        assert lines[i-1].startswith("time ")
        for l in lines[j+1:i-1]:
            _, sol, _, time = l.split()
            sol, time = float(sol), float(time)
            assert sol % 1 < 0.001 or sol % 1 > 1-0.001
            sol = round(sol)
            if len(bound_profiling_data) == 0 or sol < bound_profiling_data[-1][0]:
                bound_profiling_data.append((sol,time))
        for l in lines[i+1:]:
            k,v = l.split()
            gurobi_data[k] = v

    check_gurobi_data(gurobi_data)
    
    return bound_profiling_data, gurobi_data