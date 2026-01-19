import numpy as np
import gurobipy as gp
from gurobipy import GRB

from cg_column import Column
from exceptions import InfeasiblePricing

class Pricer:


    def __init__(self, ins, n):

        self.ins = ins
        self.n = n
        (
            self.model, self.x, 
            self.consumption_core_constraint, 
            self.consumption_bandwidth_constraint
        ) = self.create_model()


    def create_model(self):

        ins, n = self.ins, self.n
        app = ins.apps[n]

        (
            I,A,P,
            T,D,
            Q_nodes_R_core, Q_links_R_bandwidth,
            q_microservices_R_core, q_connections_R_bandwidth,
            b_microservices_zero, b_connections_zero_not_implied, b_connections_one_actual
        ) = (
            ins.I, ins.A, ins.P,
            app.T, app.D,
            ins.Q_nodes_R_core, ins.Q_links_R_bandwidth,
            app.q_microservices_R_core, app.q_connections_R_bandwidth,
            app.b_microservices_zero, app.b_connections_zero_not_implied, app.b_connections_one_actual
        )

        model = gp.Model("pricer", env=gp.Env(params={"OutputFlag" : 0}))

        x = model.addVars(T, I, vtype=GRB.BINARY, name="x")

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
            ),
            name='microservices_(u,v)_not_mappable_to_nodes_(i,j)'
        )
        
        consumption_core_constraint = model.addConstrs(
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
    
        consumption_bandwidth_constraint = model.addConstrs(
            (
                lhs[(l,m)] <= Q_links_R_bandwidth[l,m]
                for (l,m) in A
            ),
            name='on_link_(l,m)_consumption_of_resource_bandwidth'
        )
        
        return model, x, consumption_core_constraint, consumption_bandwidth_constraint


    def optimize(self, λ_positive, μ_positive, η_n):
        
        ins, n = self.ins, self.n
        app = ins.apps[n]

        (
            I,c,P,a2,
            T,D,
            q_microservices_R_core, q_connections_R_bandwidth
        ) = (
            ins.I, ins.c, ins.P, ins.a2,
            app.T, app.D,
            app.q_microservices_R_core, app.q_connections_R_bandwidth
        )

        model, x = self.model, self.x

        model.setObjective(
            gp.quicksum(c[i] * x[u,i] for u in range(T) for i in range(I)) 
            - η_n
            + gp.quicksum(
                q_microservices_R_core[u] * λ_value * x[u,i] 
                    for (i, λ_value) in λ_positive
                        for u in range(T)
            )
            + gp.quicksum(
                q_connections_R_bandwidth[u,v] * μ_value * x[u,i] * x[v,j] 
                    for ((l,m), μ_value) in μ_positive
                        for (i,j) in a2[l][m]
                            for (u,v) in D
            ), 
            GRB.MINIMIZE
        )

        model.optimize()

        if model.Status == GRB.INFEASIBLE:
            raise InfeasiblePricing()
        
        if model.Status != GRB.OPTIMAL:
            raise RuntimeError("ERROR in Pricer optimize : model.Status != GRB.OPTIMAL")

        optimal_assignment = [None] * T
        for u in range(T):
            for i in range(I):
                if x[u,i].X > 0.5:
                    # assert optimal_assignment[u] is None
                    optimal_assignment[u] = i
            # assert optimal_assignment[u] is not None

        
        new_col_cost = sum(c[optimal_assignment[u]] for u in range(T))
        
        new_col_q_core = np.zeros(I, dtype=np.int32)
        for u in range(T):
            i = optimal_assignment[u]
            new_col_q_core[i] += q_microservices_R_core[u]

        new_col_q_bandwidth = np.zeros((I, I), dtype=np.int32)
        for (u,v) in D:
            i = optimal_assignment[u]
            j = optimal_assignment[v]
            for (l,m) in P[i][j]:
                new_col_q_bandwidth[l,m] += q_connections_R_bandwidth[u,v]

        new_col = Column(
            new_col_cost, new_col_q_core, new_col_q_bandwidth, optimal_assignment
        )

        return new_col, model.ObjVal
    
    # helpers for rounding

    def fix_u_on_i(self, u, i):
        model, x = self.model, self.x
        model.addConstr(x[u,i] == 1)
        model.update()
