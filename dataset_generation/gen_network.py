import gurobipy as gp
from gurobipy import GRB
import json
import math
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
import sys

def config_validity_check(CONFIG):

    # type checks
    assert isinstance(CONFIG['network_size'],int)
    assert isinstance(CONFIG['edge_percentage'],float)
    assert isinstance(CONFIG['min_cloud-edge_connectivity_percentage'],float)
    assert isinstance(CONFIG['max_cloud-edge_connectivity_percentage'],float)
    assert isinstance(CONFIG['min_average_edge_degree'],float)
    assert isinstance(CONFIG['vehicle_min_core'],int)
    assert isinstance(CONFIG['vehicle_max_core'],int)
    assert isinstance(CONFIG['edge_min_core'],int)
    assert isinstance(CONFIG['edge_max_core'],int)
    assert isinstance(CONFIG['cloud_core'],int)
    assert isinstance(CONFIG['prob_vehicle_has_camera'],float)
    assert isinstance(CONFIG['prob_vehicle_has_gpu'],float)
    assert isinstance(CONFIG['prob_edge_has_camera'],float)
    assert isinstance(CONFIG['prob_edge_has_gpu'],float)
    assert isinstance(CONFIG['min_bandwidth'],int)
    assert isinstance(CONFIG['max_bandwidth'],int)
    assert isinstance(CONFIG['latency'],int)
    assert isinstance(CONFIG['vehicle_min_cost'],int)
    assert isinstance(CONFIG['vehicle_max_cost'],int)
    assert isinstance(CONFIG['edge_min_cost'],int)
    assert isinstance(CONFIG['edge_max_cost'],int)
    assert isinstance(CONFIG['cloud_cost'],int)
    assert isinstance(CONFIG['number_of_topologies_to_generate'],int)
    assert isinstance(CONFIG['number_of_property_resources_files_per_topology'],int)
    assert isinstance(CONFIG['save_image'],bool)
    assert isinstance(CONFIG['save_topology_files'],bool)
    assert isinstance(CONFIG['save_property_resources_files'],bool)
    assert isinstance(CONFIG['output_dir'],str)

    # value checks
    assert len(CONFIG) == 28
    assert CONFIG['network_size'] >= 10
    assert 0.0 <= CONFIG['edge_percentage'] <= 1.0
    assert 0.0 <= CONFIG['min_cloud-edge_connectivity_percentage'] <= 1.0
    assert 0.0 <= CONFIG['max_cloud-edge_connectivity_percentage'] <= 1.0
    assert CONFIG['min_cloud-edge_connectivity_percentage'] <= CONFIG['max_cloud-edge_connectivity_percentage']
    assert CONFIG['min_average_edge_degree'] >= 1.0

    assert CONFIG['vehicle_min_cost'] >= 0
    assert CONFIG['vehicle_max_cost'] >= 0
    assert CONFIG['vehicle_min_cost'] <= CONFIG['vehicle_max_cost']
    assert CONFIG['edge_min_cost'] >= 0
    assert CONFIG['edge_max_cost'] >= 0
    assert CONFIG['edge_min_cost'] <= CONFIG['edge_max_cost']
    assert CONFIG['vehicle_min_cost'] >= CONFIG['edge_max_cost']
    assert CONFIG['edge_min_cost'] >= CONFIG['cloud_cost']

    assert CONFIG['number_of_topologies_to_generate'] >= 0
    assert CONFIG['number_of_property_resources_files_per_topology'] >= 0
    assert len(CONFIG['output_dir']) > 0
    
def get_random_grid_jittered_points(count, rng):
    space_size = 1.0  # fixed 1x1 square
    side = math.ceil(math.sqrt(count))
    xs = np.linspace(0.0, 1.0, side, endpoint=False) + space_size/(2*side)
    ys = np.linspace(0.0, 1.0, side, endpoint=False) + space_size/(2*side)
    grid = np.array(np.meshgrid(xs, ys)).T.reshape(-1, 2)
    rng.shuffle(grid)
    jitter = rng.uniform(-space_size/(4*side), space_size/(4*side), size=(count, 2))
    points = grid[:count] + jitter
    return points

def solve_k_median_problem(G, k):

    V = list(G.nodes)
    SP = dict()

    for i in range(len(V)): 
        v1 = V[i]
        SP[(v1,v1)] = 0.0
        for j in range(i+1,len(V)): 
            v2 = V[j]
            SP[(v1,v2)] = nx.shortest_path_length(G,v1,v2)
            SP[(v2,v1)] = SP[(v1,v2)]
            
    model = gp.Model("k_median_problem")
    model.setParam("OutputFlag", 0)

    #   y[j] = 1 if j is selected as a center
    # x[i,j] = 1 if i is assigned to center j

    y = model.addVars(V, vtype=GRB.BINARY, name="y")
    x = model.addVars(V, V, vtype=GRB.BINARY, name="x")

    model.setObjective(
        gp.quicksum(SP[(i,j)] * x[i,j] for i in V for j in V),
        GRB.MINIMIZE
    )

    model.addConstrs(
        (gp.quicksum(x[i,j] for j in V) == 1 for i in V), 
        name="assign_each_vertex"
    )

    model.addConstrs(
        (x[i,j] <= y[j] for i in V for j in V), 
        name="assign_only_to_open_centers"
    )

    model.addConstr(
        gp.quicksum(y[j] for j in V) == k, 
        name="select_k_centers"
    )

    model.optimize()
    selected_centers = [j for j in V if y[j].X > 0.5]
    return selected_centers

def scale_values_in_range(values, new_min, new_max):

    if len(values) == 1:
        midpoint = new_min + (new_max - new_min) / 2
        return [int(round(midpoint, 0))]

    old_min = min(values)
    old_max = max(values)
    if old_min == old_max:
        midpoint = new_min + (new_max - new_min) / 2
        return [int(round(midpoint, 0)) for _ in values]

    m = (new_max - new_min) / (old_max - old_min)
    b = new_min - m * old_min
    return [int(round(m*v+b, 0)) for v in values]

def create_network_topology_file(filename, size, edges, paths, nodes_costs):
    
    with open(filename, "w") as f:
        f.write(f"{size}\n")
        f.write(' '.join([f"{i},{j}" for (i,j) in edges]) + '\n')
        f.write(' '.join(map(str,nodes_costs)) + '\n')
        for i in range(size):
            for j in range(size):
                f.write(f"{i} {j} : {' '.join(map(str,paths[i][j]))}\n")

def create_network_property_resources_file(filename, CONFIG, rng, edges, vehicle_nodes, edge_nodes):
    
    with open(filename, "w") as f:
        
        # resources/properties on nodes

        f.write("core\n")
        f.write(f"{CONFIG['cloud_core']} ")
        f.write(f"{' '.join(map(str,[rng.integers(CONFIG['edge_min_core'], CONFIG['edge_max_core']+1) for _ in edge_nodes]))} ")
        f.write(f"{' '.join(map(str,[rng.integers(CONFIG['vehicle_min_core'], CONFIG['vehicle_max_core']+1) for _ in vehicle_nodes]))}\n")

        f.write("has_camera\n")
        f.write(f"0 ")
        f.write(f"{' '.join(map(str,[1 if rng.random() < CONFIG['prob_edge_has_camera'] else 0 for _ in edge_nodes]))} ")
        f.write(f"{' '.join(map(str,[1 if rng.random() < CONFIG['prob_vehicle_has_camera'] else 0 for _ in vehicle_nodes]))}\n")

        f.write("has_gpu\n")
        f.write(f"0 ")
        f.write(f"{' '.join(map(str,[1 if rng.random() < CONFIG['prob_edge_has_gpu'] else 0 for _ in edge_nodes]))} ")
        f.write(f"{' '.join(map(str,[1 if rng.random() < CONFIG['prob_vehicle_has_gpu'] else 0 for _ in vehicle_nodes]))}\n")

        # resources/properties on edges

        f.write("bandwidth\n")
        for (i,j) in edges:
            f.write(f"{i},{j} {rng.integers(CONFIG['min_bandwidth'], CONFIG['max_bandwidth']+1)}\n")

        f.write(f"latency\n")
        for (i,j) in edges:
            f.write(f"{i},{j} {CONFIG['latency']}\n")

def generate_network(CONFIG, seed):

    euclidean_distance = lambda p1,p2: np.linalg.norm(np.array(p1) - np.array(p2))

    rng = np.random.default_rng(seed)

    I = CONFIG['network_size']

    # number of nodes by type

    cloud_count = 1                                     
    edge_count = round(CONFIG['edge_percentage'] * I)
    vehicle_count = I - edge_count - cloud_count

    # list of nodes by type

    cloud_node = 0                                   
    edge_nodes = list(range(1, edge_count+1))           
    vehicle_nodes = list(range(edge_count+1, edge_count+1+vehicle_count))

    G = nx.Graph()
    
    # place the edge nodes
    
    G.add_nodes_from(edge_nodes)
    edge_points = get_random_grid_jittered_points(edge_count, rng)
    
    assert edge_count == len(edge_nodes) == len(edge_points)

    position = dict()
    for i,node in enumerate(edge_nodes):
        position[node] = tuple(edge_points[i])

    # connect each edge node to its nearest edge neighbor

    for i in edge_nodes:
        min_dist_val = math.inf
        min_dist_node = None
        for j in edge_nodes:
            if j != i:
                d = euclidean_distance(position[i],position[j])
                if d < min_dist_val:
                    min_dist_val = d
                    min_dist_node = j
        assert min_dist_node is not None
        G.add_edge(i, min_dist_node)
    
    # make the graph connected

    while not nx.is_connected(G):
        components = list(nx.connected_components(G))
        min_dist_val = math.inf
        min_dist_pair = None
        for i in range(len(components)):
            comp = components[i]
            for j in range(i+1,len(components)):
                other_comp = components[j]
                for u in comp:
                    for v in other_comp:
                        d = euclidean_distance(position[u],position[v])
                        if d < min_dist_val:
                            min_dist_val = d
                            min_dist_pair = (u,v)
        assert min_dist_pair is not None
        G.add_edge(*min_dist_pair)

    # add edges until the average degree requirement is satisfied
    
    while np.mean([G.degree[node] for node in edge_nodes]) < CONFIG['min_average_edge_degree']:
        min_dist_val = math.inf
        min_dist_pair = None
        for i in edge_nodes:
            for j in edge_nodes:
                if j > i and not G.has_edge(i,j):
                    d = euclidean_distance(position[i],position[j])
                    if d < min_dist_val:
                        min_dist_val = d
                        min_dist_pair = (i,j)
        assert min_dist_pair is not None
        G.add_edge(*min_dist_pair)
    
    # place the cloud node and connect it to the "cloud_node_degree" medians of the graph

    cloud_node_degree = rng.integers(
        int(CONFIG['min_cloud-edge_connectivity_percentage']*edge_count), 
        int(CONFIG['max_cloud-edge_connectivity_percentage']*edge_count)+1
    )

    medians = solve_k_median_problem(G, cloud_node_degree)

    G.add_node(cloud_node)
    cloud_point = (1/2, 1/2)
    position[cloud_node] = cloud_point
    for node in medians:
        G.add_edge(cloud_node, node)

    # place the vehicle nodes
    
    G.add_nodes_from(vehicle_nodes)
    vehicle_points = get_random_grid_jittered_points(vehicle_count, rng)
    
    assert vehicle_count == len(vehicle_nodes) == len(vehicle_points)

    for i,node in enumerate(vehicle_nodes):
        position[node] = tuple(vehicle_points[i])

    # connect each vehicle node to its nearest edge neighbor

    for i in vehicle_nodes:
        min_dist_val = math.inf
        min_dist_node = None
        for j in edge_nodes:
            if j != i:
                d = euclidean_distance(position[i],position[j])
                if d < min_dist_val:
                    min_dist_val = d
                    min_dist_node = j
        assert min_dist_node is not None
        G.add_edge(i, min_dist_node)

    edges = sorted([(min(i,j),max(i,j)) for (i,j) in G.edges]) 

    # setting communication path for each node pair

    paths = [[None] * I for _ in range(I)]
    for i in range(I):
        for j in range(I):
            if i <= j:  
                path = list(map(int,list(nx.all_shortest_paths(G,i,j))[0]))
            else:       
                path = list(map(int,list(nx.all_shortest_paths(G,i,j))[-1]))
            paths[i][j] = path
            assert paths[i][j][0] == i and paths[i][j][-1] == j

    # setting node costs

    vehicle_nodes_costs = scale_values_in_range(
        [round(v,2) for _,v in nx.harmonic_centrality(G,vehicle_nodes).items()], 
        CONFIG['vehicle_min_cost'], CONFIG['vehicle_max_cost'])

    edge_nodes_costs = scale_values_in_range(
        [round(v,2) for _,v in nx.harmonic_centrality(G,edge_nodes).items()], 
        CONFIG['edge_min_cost'], CONFIG['edge_max_cost'])

    cloud_nodes_costs = [CONFIG['cloud_cost']]

    nodes_costs = cloud_nodes_costs + edge_nodes_costs + vehicle_nodes_costs

    # output operations

    if (CONFIG['save_image'] or CONFIG['save_topology_files'] or CONFIG['save_property_resources_files']) and not os.path.exists(CONFIG['output_dir']):
        os.makedirs(CONFIG['output_dir'])

    if CONFIG['save_image']:
        network_image_filename = os.path.join(CONFIG['output_dir'], f"n_{I}_s_{seed}.png")
        plt.figure(figsize=(6, 6))
        nx.draw_networkx_nodes(G, pos=position, nodelist=edge_nodes, node_color="yellow", node_size=100, edgecolors="black")
        nx.draw_networkx_nodes(G, pos=position, nodelist=[cloud_node], node_color="red", node_size=100, edgecolors="black")
        nx.draw_networkx_nodes(G, pos=position, nodelist=vehicle_nodes, node_color="green", node_size=100, edgecolors="black")
        nx.draw_networkx_edges(G, pos=position, edge_color="gray")
        plt.axis("off")
        plt.savefig(network_image_filename, bbox_inches="tight", dpi=300)
        plt.close()

    if CONFIG['save_topology_files']:
        topology_filename = os.path.join(CONFIG['output_dir'], f"n_{I}_s_{seed}.dat")
        create_network_topology_file(
            topology_filename, I, edges, paths, nodes_costs)

    if CONFIG['save_property_resources_files']:
        for i in range(CONFIG['number_of_property_resources_files_per_topology']):
            property_resources_filename = os.path.join(CONFIG['output_dir'], f"n_{I}_s_{seed}_rp_{i}.dat")
            create_network_property_resources_file(
                property_resources_filename, CONFIG, rng, edges, vehicle_nodes, edge_nodes)

if __name__ == '__main__':

    with open("config_gen_network.json", "r") as f:
        CONFIG = json.load(f)

    assert len(sys.argv) == 2
    CONFIG['network_size'] = int(sys.argv[1])

    config_validity_check(CONFIG)

    for seed in range(CONFIG['number_of_topologies_to_generate']):
        generate_network(CONFIG, seed)