import numpy as np
from copy import deepcopy

from cg_column import Column

EPSILON = 0.001

class PricerLocalSearch:


    def __init__(self, ins, n):
        
        self.ins = ins
        self.n = n
        app = self.ins.apps[n]
        
        self.incoming = [None] * app.T
        self.outcoming = [None] * app.T
        for u in range(app.T):
            self.incoming[u] = set()
            self.outcoming[u] = set()
        for u,v in app.D:
            self.outcoming[u].add(v)
            self.incoming[v].add(u)

        self.feasible_nodes = [app.b_microservices_one[u] for u in range(app.T)]
        self.movable_microservices = [u for u in range(app.T) if len(self.feasible_nodes[u]) > 1]
        self.c_adjusted = np.full((app.T,self.ins.I),0.0)


    def remove_from_movable_microservices(self, u):

        feasible_nodes, movable_microservices = self.feasible_nodes, self.movable_microservices

        
        if u in movable_microservices:
            movable_microservices.remove(u)
        # else:
        #     assert len(feasible_nodes[u]) == 1


    def remove_microservice_from_node(self, u, i, cur_solution, cur_z, used_core, used_bandwidth, μ):

        ins, n = self.ins, self.n
        app = ins.apps[n]
        P, q_microservices_R_core, q_connections_R_bandwidth = ins.P, app.q_microservices_R_core, app.q_connections_R_bandwidth
        incoming, outcoming, c_adjusted = self.incoming, self.outcoming, self.c_adjusted


        i = cur_solution[u]

        # update objective with u removed
        cur_z -= c_adjusted[u][i]

        # update resources with u removed
        used_core[i] -= q_microservices_R_core[u]
        for v in incoming[u]:
            j = cur_solution[v]
            for (l,m) in P[j][i]:
                cur_z -= q_connections_R_bandwidth[v,u] * μ[l,m]
                used_bandwidth[l,m] -= q_connections_R_bandwidth[v,u]
        for v in outcoming[u]:
            j = cur_solution[v]
            for (l,m) in P[i][j]:
                cur_z -= q_connections_R_bandwidth[u,v] * μ[l,m]
                used_bandwidth[l,m] -= q_connections_R_bandwidth[u,v]

        cur_solution[u] = None
        return cur_z
    
    """
    def is_feasible(
        self, cur_solution, used_core, used_bandwidth
    ):
        (
            I,A,c,P,a2,
            T,D,
            Q_nodes_R_core, Q_links_R_bandwidth,
            q_microservices_R_core, q_connections_R_bandwidth,
            b_microservices_zero, b_microservices_one, 
            b_connections_zero_not_implied, b_connections_one, b_connections_one_actual
        ) = (
            self.I,self.A,self.c,self.P,self.a2,
            self.T,self.D,
            self.Q_nodes_R_core, self.Q_links_R_bandwidth,
            self.q_microservices_R_core, self.q_connections_R_bandwidth,
            self.b_microservices_zero, self.b_microservices_one, 
            self.b_connections_zero_not_implied, self.b_connections_one, self.b_connections_one_actual
        )

        # incoming = self.incoming
        # outcoming = self.outcoming
        
        assert None not in cur_solution
        
        for i in range(I):
            if used_core[i] > Q_nodes_R_core[i]:
                return False
        
        for (l,m) in A:
            if used_bandwidth[l,m] > Q_links_R_bandwidth[l,m]:
                return False

        return True
    """

    def is_feasible_improved(self, cur_solution, used_core, used_bandwidth, u, j):
        
        ins = self.ins
        P, Q_nodes_R_core, Q_links_R_bandwidth = ins.P, ins.Q_nodes_R_core, ins.Q_links_R_bandwidth
        incoming, outcoming = self.incoming, self.outcoming

        
        # assert None not in cur_solution
        
        if used_core[j] > Q_nodes_R_core[j]:
            return False
        
        for v in incoming[u]:
            j_ = cur_solution[v]
            for (l,m) in P[j_][j]:
                if used_bandwidth[l,m] > Q_links_R_bandwidth[l,m]:
                    return False
        for v in outcoming[u]:
            j_ = cur_solution[v]
            for (l,m) in P[j][j_]:
                if used_bandwidth[l,m] > Q_links_R_bandwidth[l,m]:
                    return False

        return True
    

    # assumes that u is currently not assigned
    def assign_microservice_to_node(self, u, i, cur_solution, cur_z, used_core, used_bandwidth, μ):
        
        ins, n = self.ins, self.n
        app = ins.apps[n]
        P, q_microservices_R_core, q_connections_R_bandwidth = ins.P, app.q_microservices_R_core, app.q_connections_R_bandwidth
        incoming, outcoming, c_adjusted = self.incoming, self.outcoming, self.c_adjusted


        # assert cur_solution[u] is None

        # update objective with u inserted
        cur_z += c_adjusted[u][i]

        # update resources with u inserted
        used_core[i] += q_microservices_R_core[u]
        for v in incoming[u]:
            j = cur_solution[v]
            for (l,m) in P[j][i]:
                cur_z += q_connections_R_bandwidth[v,u] * μ[l,m]
                used_bandwidth[l,m] += q_connections_R_bandwidth[v,u]
        for v in outcoming[u]:
            j = cur_solution[v]
            for (l,m) in P[i][j]:
                cur_z += q_connections_R_bandwidth[u,v] * μ[l,m]
                used_bandwidth[l,m] += q_connections_R_bandwidth[u,v]

        cur_solution[u] = i
        return cur_z
    

    def optimize(self, λ_positive, μ, η_n, cur_column, fixings=None):

        ins, n = self.ins, self.n
        app = ins.apps[n]
        (
            I, c, P, T, D, 
            q_microservices_R_core, q_connections_R_bandwidth, b_connections_zero_not_implied
        ) = (
            ins.I, ins.c, ins.P, app.T, app.D, 
            app.q_microservices_R_core, app.q_connections_R_bandwidth, app.b_connections_zero_not_implied
        )
        incoming, outcoming, c_adjusted = self.incoming, self.outcoming, self.c_adjusted
        feasible_nodes, movable_microservices = self.feasible_nodes, self.movable_microservices


        # if fixings is not None:
        #     for (u,i) in fixings:
        #         assert u not in movable_microservices
        #         assert cur_column.original_x[u] == i

        used_core = deepcopy(cur_column.col_q_core)
        used_bandwidth = deepcopy(cur_column.col_q_bandwidth)

        for u in range(T):

            for i in range(I):
                c_adjusted[u][i] = c[i]

            for (i, λ_value) in λ_positive:
                c_adjusted[u][i] += q_microservices_R_core[u] * λ_value

        cur_solution = cur_column.original_x[:]
        cur_z = -η_n

        for u,i in enumerate(cur_solution):
            cur_z += c_adjusted[u][i]
        
        for (u,v) in D:
            i,j = cur_solution[u], cur_solution[v]
            for (l,m) in P[i][j]:
                cur_z += q_connections_R_bandwidth[u,v] * μ[l,m]
        
        # assert cur_z >= -EPSILON

        # print(f"local search starting cur_z {cur_z}")

        iteration = 0
        while True:

            iteration += 1

            # print()
            # print(f"iteration {iteration} ----------------------------------------")
            # print()

            ### - trovo la mossa u*,j* che massimizza G(u,j)

            G_best, G_u, G_j = 0, None, None

            base_cost = cur_z
            for u in movable_microservices:
                
                # assert abs(cur_z - base_cost) < EPSILON
                cur_z = base_cost

                i = cur_solution[u]

                cur_z = self.remove_microservice_from_node(u, i, cur_solution, cur_z, used_core, used_bandwidth, μ)
                base_cost_without_u = cur_z

                u_removal_prize = base_cost - cur_z
                # assert u_removal_prize >= 0

                # print(f"\tprize of removing {u} from cur_sol : {u_removal_prize}")
                
                set_feasible_nodes_u = set(feasible_nodes[u])
                # print(f"\t# feasible nodes for u : {len(set_feasible_nodes_u)}")
                for j in feasible_nodes[u]:
                    for v in incoming[u]:
                        k = cur_solution[v]
                        if (k,j) in b_connections_zero_not_implied[v][u]:
                            if j in set_feasible_nodes_u:
                                set_feasible_nodes_u.remove(j)
                    for v in outcoming[u]:
                        k = cur_solution[v]
                        if (j,k) in b_connections_zero_not_implied[u][v]:
                            if j in set_feasible_nodes_u:
                                set_feasible_nodes_u.remove(j)
                # print(f"\t# feasible nodes for u according to b: {len(set_feasible_nodes_u)}")

                for j in set_feasible_nodes_u:
                    if j != i:

                        cur_z = self.assign_microservice_to_node(u, j, cur_solution, cur_z, used_core, used_bandwidth, μ)

                        # il controllo di essere in una situazione feasible puo' essere fatto in modo piu intelligente 
                        # in base a cio che è stato veramente toccato dalla funzione prima
                        # if self.is_feasible(cur_solution, used_core, used_bandwidth):
                        
                        f1 = self.is_feasible_improved(cur_solution, used_core, used_bandwidth, u, j)
                        #f2 = self.is_feasible(cur_solution, used_core, used_bandwidth)
                        # assert f1 == f2
                        if f1:
                            
                            u_swap_prize = base_cost - cur_z
                            if u_swap_prize > G_best:
                                G_best = u_swap_prize
                                G_u = u
                                G_j = j
                                # print(f"\t\tfound new best swap : {u} from {i} to {j} with gain {u_swap_prize}")

                        cur_z = self.remove_microservice_from_node(u, j, cur_solution, cur_z, used_core, used_bandwidth, μ)
                        # assert abs(cur_z - base_cost_without_u) < EPSILON
                        cur_z = base_cost_without_u

                cur_z = self.assign_microservice_to_node(u, i, cur_solution, cur_z, used_core, used_bandwidth, μ)


            ### - IF G(u*,j*) <= 0 : BREAK

            if G_best == 0:
                # assert G_u == None and G_j == None 
                break
            
            ### - aggiorno cur_sol facendo la mossa u*,j*

            G_i = cur_solution[G_u]

            cur_z = self.remove_microservice_from_node(G_u, G_i, cur_solution, cur_z, used_core, used_bandwidth, μ)
            cur_z = self.assign_microservice_to_node(G_u, G_j, cur_solution, cur_z, used_core, used_bandwidth, μ)

        # building the column
        new_col = None
        if cur_z < -EPSILON:
        
            new_col_q_core = np.zeros(I, dtype=np.int32)
            for u in range(T):
                i = cur_solution[u]
                new_col_q_core[i] += q_microservices_R_core[u]

            new_col_q_bandwidth = np.zeros((I, I), dtype=np.int32)
            for (u,v) in D:
                i = cur_solution[u]
                j = cur_solution[v]
                for (l,m) in P[i][j]:
                    new_col_q_bandwidth[l,m] += q_connections_R_bandwidth[u,v]

            new_col = Column(
                sum(c[cur_solution[u]] for u in range(T)),  # col_cost
                new_col_q_core,                             # col_q_core
                new_col_q_bandwidth,                        # col_q_bandwidth 
                deepcopy(cur_solution)                      # original_x
            )

        # end = perf_counter()
        # print(f"local search done in {end-start:.2f} s , {cur_z}")
        # print(f"local search ending cur_z {cur_z}")
        # print(f"local search #iter {iteration}")
        
        return new_col, cur_z