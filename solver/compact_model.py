import gurobipy as gp
from gurobipy import GRB
from time import perf_counter
import os

def build_compact_model(ins):

    (   
        I, A, c, P, T, D, 
        Q_nodes_R_core, q_microservices_R_core,
        Q_links_R_bandwidth, q_connections_R_bandwidth,
        b_microservices_zero, b_connections_zero_not_implied, b_connections_one_actual
    ) = (
        ins.I, ins.A, ins.c, ins.P, ins.app_merged.T, ins.app_merged.D,
        ins.Q_nodes_R_core, ins.app_merged.q_microservices_R_core,
        ins.Q_links_R_bandwidth, ins.app_merged.q_connections_R_bandwidth,
        ins.app_merged.b_microservices_zero, ins.app_merged.b_connections_zero_not_implied, ins.app_merged.b_connections_one_actual
    )

    model = gp.Model("CAVIA_compact_model")
    
    x = model.addVars(T, I, vtype=GRB.BINARY, name = "x")

    model.setObjective(
        gp.quicksum(c[i] * x[u,i] for u in range(T) for i in range(I)), 
        GRB.MINIMIZE
    )

    model.addConstrs(
        (
            gp.quicksum(x[u,i] for i in range(I)) == 1 
            for u in range(T)
        ), 
        name='microservice_u_mapped_to_exactly_one_node'
    )

    model.addConstrs(
        (
            x[u,i] == 0 
            for u in range(T) 
                for i in b_microservices_zero[u]
        ), 
        name='microservice_u_not_mappable_to_node_i'
    )
    
    model.addConstrs(
        (
            x[u,i] + x[v,j] <= 1
            for (u,v) in D
                for (i,j) in b_connections_zero_not_implied[u][v]
                # for (i,j) in b_connections_zero[u][v] 
        ),
        name='microservices_(u,v)_not_mappable_to_nodes_(i,j)'
    )

    model.addConstrs(
        (
            gp.quicksum(q_microservices_R_core[u] * x[u,i] for u in range(T)) <= Q_nodes_R_core[i]
            for i in range(I)
        ),
        name='on_node_i_consumption_of_resource_core'
    )

    lhs = dict()
    for (l,m) in A:
        lhs[(l,m)] = gp.QuadExpr(0)
        
    for (u,v) in D:
        for (i,j) in b_connections_one_actual[u][v]:
            for (l,m) in P[i][j]:
                lhs[(l,m)].add(q_connections_R_bandwidth[u,v] * x[u,i] * x[v,j])

    model.addConstrs(
        (
            lhs[(l,m)] <= Q_links_R_bandwidth[l,m]
            for (l,m) in A
        ),
        name='on_link_(l,m)_consumption_of_resource_bandwidth'
    )
    
    return model, x
        

def optimize_compact_model(
    model, time_limit_seconds, logfile_custom, logfile_grb, opening_mode='w'
):

    # model.setParam("OutputFlag", 0) 
    model.setParam("LogToConsole", 0) 
    if os.path.exists(logfile_grb): 
        os.remove(logfile_grb)
    model.setParam("LogFile", logfile_grb)
    model.setParam('TimeLimit', time_limit_seconds)
    model.update()

    with open(logfile_custom, opening_mode) as f:
        start_model = perf_counter()
        def callback(model, where):
            if where == GRB.Callback.MIPSOL:
                obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                t = perf_counter() - start_model
                f.write(f"found_sol {obj} time {t:.3f}\n")
        model.optimize(callback)
        end_model = perf_counter()
        f.write(f"time {end_model-start_model:.3f}\n")
        f.write("GUROBI\n")
        assert model.Status in (2,3,9)
        status_to_str = {
            2: 'OPTIMAL',
            3: 'INFEASIBLE',
            9: 'TIME_LIMIT',
        }

        output = {
            "Status" : status_to_str[model.Status],
            "Runtime" : model.Runtime,
            "NodeCount" : model.NodeCount,
            "IterCount" : model.IterCount,
            "NumVars" : model.NumVars,
            "NumConstrs" : model.NumConstrs
        }

        try:                    output["ObjBound"] = model.ObjBound
        except Exception as _:  output["ObjBound"] = None

        try:                    output["ObjBoundC"] = model.ObjBoundC
        except Exception as _:  output["ObjBoundC"] = None
        
        try:                    output["ObjVal"] = model.ObjVal
        except Exception as _:  output["ObjVal"] = None

        try:                    output["MIPGap"] = model.MIPGap
        except Exception as _:  output["MIPGap"] = None

        for k in [
            "Status", "Runtime", "NodeCount", "IterCount", "NumVars", "NumConstrs",
            "ObjBound", "ObjBoundC", "ObjVal", "MIPGap",
        ]:
            f.write(f"{k} {output[k]}\n")

    return output

def optimize_compact_model_restricted(
    model, time_limit_seconds, logfile_custom, logfile_grb,
    fixings, instance, x
):

    with open(logfile_custom, 'w') as f:
        f.write(f"FIXINGS_START\n")
        for assignment in fixings:
            f.write(f"{assignment}\n")
        f.write(f"FIXINGS_END\n")
        
    # adding fixing constraints
    N = len(instance.apps)
    shift = [0] * N
    for i in range(1,N):
        shift[i] = shift[i-1] + (instance.apps[i-1]).T
    
    fixing_constraints = []
    for (n,u,i) in fixings:
        c = model.addConstr(
            x[u+shift[n],i] == 1, 
            name=f'fixed_app_{n}_microservice_{u}'
        )
        fixing_constraints.append(c)

    # output = optimize_compact_model(
    #     model, time_limit_seconds, logfile_custom, logfile_grb, opening_mode='a'
    # )
    
    # model.setParam("OutputFlag", 0) 
    model.setParam("LogToConsole", 0) 
    if os.path.exists(logfile_grb): 
        os.remove(logfile_grb)
    model.setParam("LogFile", logfile_grb)
    model.setParam('TimeLimit', time_limit_seconds)
    model.update()

    with open(logfile_custom, 'a') as f:
        start_model = perf_counter()
        def callback(model, where):
            if where == GRB.Callback.MIPSOL:
                obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                t = perf_counter() - start_model
                f.write(f"found_sol {obj} time {t:.3f}\n")
        model.optimize(callback)
        end_model = perf_counter()
        f.write(f"time {end_model-start_model:.3f}\n")
        f.write("GUROBI\n")
        assert model.Status in (2,3,9)
        status_to_str = {
            2: 'OPTIMAL',
            3: 'INFEASIBLE',
            9: 'TIME_LIMIT',
        }

        output = {
            "Status" : status_to_str[model.Status],
            "Runtime" : model.Runtime,
            "NodeCount" : model.NodeCount,
            "IterCount" : model.IterCount,
            "NumVars" : model.NumVars,
            "NumConstrs" : model.NumConstrs
        }

        try:                    output["ObjBound"] = model.ObjBound
        except Exception as _:  output["ObjBound"] = None

        try:                    output["ObjBoundC"] = model.ObjBoundC
        except Exception as _:  output["ObjBoundC"] = None
        
        try:                    output["ObjVal"] = model.ObjVal
        except Exception as _:  output["ObjVal"] = None

        try:                    output["MIPGap"] = model.MIPGap
        except Exception as _:  output["MIPGap"] = None

        for k in [
            "Status", "Runtime", "NodeCount", "IterCount", "NumVars", "NumConstrs",
            "ObjBound", "ObjBoundC", "ObjVal", "MIPGap",
        ]:
            f.write(f"{k} {output[k]}\n")

    # remove fixing constraints
    for c in fixing_constraints: 
        model.remove(c) 
    model.update()

    return output
