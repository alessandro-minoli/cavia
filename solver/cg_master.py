import numpy as np
from math import ceil
import gurobipy as gp
from gurobipy import GRB

from cg_column import Column

import random
random.seed(42)

EPSILON = 0.001

class Master:

    
    def __init__(self, ins):

        self.ins = ins

        self.COLUMNS_POOL = self.create_and_initialize_columns_pool()
        self.model, self.z = self.create_model()

        I = self.ins.I
        self.μ = np.zeros((I, I), dtype=np.float32)
    
    
    def create_and_initialize_columns_pool(self):

        I,c,N,apps = self.ins.I, self.ins.c, len(self.ins.apps), self.ins.apps
        
        COLUMNS_POOL = [[] for _ in range(N)]
        
        max_node_cost = ceil(np.max(c))+1
        for n in range(N): 
            
            dummy = Column(
                col_cost = apps[n].T * max_node_cost, 
                col_q_core = np.zeros(I, dtype=np.int32), 
                col_q_bandwidth = np.zeros((I, I), dtype=np.int32), 
                original_x = [None] * apps[n].T
            )

            COLUMNS_POOL[n].append(dummy)

        return COLUMNS_POOL
    
    
    def create_model(self):

        COLUMNS_POOL = self.COLUMNS_POOL
        
        (
            I,A,N,
            Q_nodes_R_core,Q_links_R_bandwidth
        ) = (
            self.ins.I, self.ins.A, len(self.ins.apps), 
            self.ins.Q_nodes_R_core, self.ins.Q_links_R_bandwidth
        )

        model = gp.Model("master", env=gp.Env(params={"OutputFlag" : 0}))

        z = [[None] * (len(COLUMNS_POOL[n])) for n in range(N)]
        for n in range(N): 
            for p in range(len(COLUMNS_POOL[n])):
                z[n][p] = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0)
                
        model.setObjective(
            sum(
                COLUMNS_POOL[n][p].col_cost * z[n][p]
                for n in range(N) 
                    for p in range(len(COLUMNS_POOL[n]))
            ), 
            GRB.MINIMIZE
        )

        for i in range(I):
            model.addConstr(
                - sum(
                    COLUMNS_POOL[n][p].col_q_core[i] * z[n][p]
                    for n in range(N) 
                        for p in range(len(COLUMNS_POOL[n]))
                ) >= - Q_nodes_R_core[i], 
                name = f"on_node_{i}_consumption_of_core"
            )

        for (l,m) in A:
            model.addConstr(
                - sum(
                    COLUMNS_POOL[n][p].col_q_bandwidth[l,m] * z[n][p]
                    for n in range(N) 
                        for p in range(len(COLUMNS_POOL[n]))
                ) >= - Q_links_R_bandwidth[l,m],
                name = f"on_link_({l},{m})_consumption_of_bandwidth"
            )

        for n in range(N):
            model.addConstr(
                    sum(z[n][p] for p in range(len(COLUMNS_POOL[n]))) >= 1,
                    name = f"convexity_{n}"
                )
        
        model.update()

        return model,z

    
    def add_column(self, n, new_column, checked=False):

        I, A = self.ins.I, self.ins.A
        COLUMNS_POOL, model, z = self.COLUMNS_POOL, self.model, self.z
        
        if checked and new_column in COLUMNS_POOL[n]:
            raise RuntimeError(f"ERROR in Master add_column : added duplicate column for {n}-th app")
    
        COLUMNS_POOL[n].append(new_column)

        z[n].append(model.addVar(vtype=GRB.CONTINUOUS, lb=0.0))
        p = len(z[n])-1

        z[n][p].Obj = new_column.col_cost

        for i in range(I):
            model.chgCoeff(
                model.getConstrByName(f"on_node_{i}_consumption_of_core"), 
                z[n][p], 
                -new_column.col_q_core[i]
            )

        for (l,m) in A:
            model.chgCoeff(
                model.getConstrByName(f"on_link_({l},{m})_consumption_of_bandwidth"), 
                z[n][p], 
                -new_column.col_q_bandwidth[l,m]
            )

        model.chgCoeff(
            model.getConstrByName(f"convexity_{n}"), 
            z[n][p], 
            1
        )

    
    def optimize(self):

        # assert self.model is not None

        I,A,N = self.ins.I, self.ins.A, len(self.ins.apps)

        model, μ = self.model, self.μ

        model.optimize()

        if model.Status != GRB.OPTIMAL:
            raise RuntimeError(f"ERROR in Master optimize : model.Status != GRB.OPTIMAL")
        
        # retrieving dual values

        λ_positive = []
        for i in range(I):
            value = model.getConstrByName(f"on_node_{i}_consumption_of_core").Pi
            # assert value >= -EPSILON
            value = max(0, value)
            if value > 0:
                λ_positive.append((i,value))
        λ_positive = tuple(λ_positive)

        μ_positive = []
        for (l,m) in A:
            value = model.getConstrByName(f"on_link_({l},{m})_consumption_of_bandwidth").Pi
            # assert value >= -EPSILON
            value = max(0, value)
            μ[l][m] = value
            if value > 0:
                μ_positive.append(((l,m),value))
        μ_positive = tuple(μ_positive)
            
        η = []
        for n in range(N):
            value = model.getConstrByName(f"convexity_{n}").Pi
            # assert value >= -EPSILON
            value = max(0, value)
            η.append(value)
        η = tuple(η)

        return λ_positive, μ_positive, μ, η, model.ObjVal
    
    def get_most_recent_mapping_of_app(self, n):
        return self.COLUMNS_POOL[n][-1]
    
    def still_using_dummy(self):
        N, z = len(self.ins.apps), self.z
        return any(z[n][0].X > EPSILON for n in range(N))

    def solution_is_integer(self):
        COLUMNS_POOL, N, z = self.COLUMNS_POOL, len(self.ins.apps), self.z
        for n in range(N): 
            if not any(z[n][p].X >= 0.999 for p in range(len(COLUMNS_POOL[n]))):
                return False
        return True
    
    ### helpers for rounding
    
    def get_most_selected_fixing(self, still_to_fix):

        COLUMNS_POOL, z, N = self.COLUMNS_POOL, self.z, len(self.ins.apps)
        
        shuffled_N = list(range(N))
        random.shuffle(shuffled_N)

        n_most, u_most, i_most, z_most = -1, -1, -1, -1

        for n in shuffled_N:

            value = {u : dict() for u in still_to_fix[n]}

            for p in range(len(COLUMNS_POOL[n])):
                if z[n][p].X > 0: 
                    col = COLUMNS_POOL[n][p]
                    for u,i in enumerate(col.original_x):
                        if u in still_to_fix[n]:
                            if i not in value[u]:   
                                value[u][i] = z[n][p].X
                            else:                   
                                value[u][i] += z[n][p].X
                            if value[u][i] > z_most:
                                n_most, u_most, i_most, z_most = n, u, i, value[u][i]
        
        # assert (n_most, u_most, i_most, z_most) != (-1, -1, -1, -1)

        return n_most, u_most, i_most, z_most
        
    def remove_columns_of_n_in_which_u_is_not_on_i(self, n, u, i):

        COLUMNS_POOL, z, model = self.COLUMNS_POOL, self.z, self.model

        to_remove = []
        removed_columns_count = 0
        for p in range(1,len(COLUMNS_POOL[n])):
            if COLUMNS_POOL[n][p].original_x[u] != i:
                to_remove.append(p)
                removed_columns_count += 1

        for p in to_remove:
            model.remove(z[n][p])
            z[n][p] = None
            COLUMNS_POOL[n][p] = None

        model.update()
        
        COLUMNS_POOL[n] = list(filter(lambda x: x is not None, COLUMNS_POOL[n]))
        z[n] = list(filter(lambda x: x is not None, z[n]))

        # assert COLUMNS_POOL[n] is self.COLUMNS_POOL[n]
        # assert z[n] is self.z[n]

        return removed_columns_count
