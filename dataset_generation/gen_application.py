import json
import math
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
from scipy.stats import truncnorm
import sys

def config_validity_check(CONFIG):

    # type checks
    assert isinstance(CONFIG['app_size'],int)
    assert CONFIG['app_topology'] in ('path','tree','graph')
    assert isinstance(CONFIG['edge_percentage'],float)
    assert isinstance(CONFIG['vehicle_min_core'],int)
    assert isinstance(CONFIG['vehicle_max_core'],int)
    assert isinstance(CONFIG['edge_min_core'],int)
    assert isinstance(CONFIG['edge_max_core'],int)
    assert isinstance(CONFIG['cloud_min_core'],int)
    assert isinstance(CONFIG['cloud_max_core'],int)
    assert isinstance(CONFIG['prob_vehicle_needs_camera'],float)
    assert isinstance(CONFIG['prob_vehicle_needs_gpu'],float)
    assert isinstance(CONFIG['prob_edge_needs_camera'],float)
    assert isinstance(CONFIG['prob_edge_needs_gpu'],float)
    assert isinstance(CONFIG['min_bandwidth'],int)
    assert isinstance(CONFIG['max_bandwidth'],int)
    assert isinstance(CONFIG['latency'],list)
    assert all(isinstance(val,int) for val in CONFIG['latency'])
    assert isinstance(CONFIG['number_of_topologies_to_generate'],int)
    assert isinstance(CONFIG['number_of_property_resources_files_per_topology'],int)
    assert isinstance(CONFIG['save_image'],bool)
    assert isinstance(CONFIG['save_topology_files'],bool)
    assert isinstance(CONFIG['save_property_resources_files'],bool)
    assert isinstance(CONFIG['output_dir'],str)

    # value checks
    assert len(CONFIG) == 22
    assert 3 <= CONFIG['app_size'] <= 15
    assert 0.0 <= CONFIG['edge_percentage'] <= 1.0

    assert CONFIG['number_of_topologies_to_generate'] >= 0
    assert CONFIG['number_of_property_resources_files_per_topology'] >= 0
    assert len(CONFIG['output_dir']) > 0

def get_path_topology(size):
    G = nx.DiGraph()
    G.add_nodes_from(range(size))
    for i in range(size-1):
        G.add_edge(i,i+1)
    return G

def get_random_tree_topology(size,rng):
    G = nx.DiGraph()
    G.add_nodes_from(range(size))
    for i in range(1,size):
        j = rng.integers(0,i)
        G.add_edge(i,j)
    return G

def generate_graph_from_tree(size,rng,G):
    
    T = size

    mean, std, low, upp = round((1/3)*T), T/3, 1, T
    truncated_normal_rv = truncnorm((low-mean)/std, (upp-mean)/std, loc=mean, scale=std)

    G_complement = nx.complement(G)
    G_complement_edges = list(G_complement.edges)

    assert len(G.edges) + len(G_complement_edges) == T * (T-1)

    # vogliamo che al peggio il # archi totale aumenti di T, cioè che praticamente raddoppi.
    # il numero di archi aggiunti è una variabile casuale

    number_of_arcs_to_add = round(truncated_normal_rv.rvs()) 
    assert 1 <= number_of_arcs_to_add <= T 
    number_of_arcs_to_add = min(number_of_arcs_to_add, len(G_complement_edges))
    assert number_of_arcs_to_add != 0
        
    arcs_to_add = rng.choice(G_complement_edges, size=number_of_arcs_to_add, replace=False)
    for u,v in arcs_to_add:
        G.add_edge(u,v)

    return G

def generate_application(CONFIG, seed):

    rng = np.random.default_rng(seed)

    T = CONFIG['app_size']

    # regardless of edge_percentage, 
    # vehicle_count and cloud_count must be > 0
    edge_count = math.ceil((T-2) * CONFIG['edge_percentage'])
    vehicle_count = math.ceil((T-edge_count)/2)
    cloud_count = T-edge_count-vehicle_count

    assert edge_count >= vehicle_count
    assert edge_count >= cloud_count
    assert vehicle_count >= cloud_count

    if CONFIG['app_topology'] == "path":
        
        G = get_path_topology(T)

        vehicle = set(range(vehicle_count))
        edge = set(range(vehicle_count,vehicle_count+edge_count))
        cloud = set(range(vehicle_count+edge_count,T))

    else:
        # CONFIG['app_topology'] in ("tree", "graph")
        
        G = get_random_tree_topology(T,rng)

        nodes_sorted_by_dist_from_0 = sorted(list(G.nodes),key=lambda x: nx.shortest_path_length(G,x,0), reverse=True)
        vehicle = set(nodes_sorted_by_dist_from_0[:vehicle_count])
        edge = set(nodes_sorted_by_dist_from_0[vehicle_count:vehicle_count+edge_count])
        cloud = set(nodes_sorted_by_dist_from_0[vehicle_count+edge_count:])

        if CONFIG['app_topology'] == "graph":
            G = generate_graph_from_tree(T,rng,G)

    assert vehicle | edge | cloud == set(range(T))
    assert vehicle & edge & cloud == set()
    assert len(vehicle) == vehicle_count
    assert len(edge) == edge_count
    assert len(cloud) == cloud_count

    edges = sorted(list(G.edges())) 

    # output operations

    if (CONFIG['save_image'] or CONFIG['save_topology_files'] or CONFIG['save_property_resources_files']) and not os.path.exists(CONFIG['output_dir']):
        os.makedirs(CONFIG['output_dir'])

    if CONFIG['save_image']:
        app_image_filename = os.path.join(CONFIG['output_dir'], f"{CONFIG['app_topology']}_a_{T}_s_{seed}_.png")
        colors = ["green" if i in vehicle else ("yellow" if i in edge else "red") for i in range(T)]
        plt.figure(figsize=(6, 6))
        if CONFIG['app_topology'] == "path":
            nx.draw(G, pos = nx.circular_layout(G), with_labels=False, node_color=colors, edgecolors="black")
        elif CONFIG['app_topology'] == "tree":
            nx.draw(G, pos=nx.planar_layout(G), with_labels=False, node_color=[colors[node] for node in G.nodes], edgecolors="black")
        else:
            nx.draw(G, pos=nx.spring_layout(G), with_labels=False, node_color=[colors[node] for node in G.nodes], edgecolors="black")
        plt.savefig(app_image_filename, bbox_inches="tight", dpi=300)
        plt.close()

    if CONFIG['save_topology_files']:
        topology_filename = os.path.join(CONFIG['output_dir'], f"{CONFIG['app_topology']}_a_{T}_s_{seed}.dat")
        create_application_topology_file(topology_filename, T, edges)

    if CONFIG['save_property_resources_files']:
        for i in range(CONFIG['number_of_property_resources_files_per_topology']):
            property_resources_filename = os.path.join(CONFIG['output_dir'], f"{CONFIG['app_topology']}_a_{T}_s_{seed}_rp_{i}.dat")
            create_application_property_resources_file(
                property_resources_filename, CONFIG, rng, T, edges, vehicle, edge, cloud)

def create_application_topology_file(filename, T, edges):
    with open(filename, "w") as f:
        f.write(f"{T}\n")
        f.write(" ".join([f"{u},{v}" for (u,v) in edges]) + "\n")

def create_application_property_resources_file(filename, CONFIG, rng, T, edges, vehicle, edge, cloud):

    with open(filename, "w") as f:

        # resources/properties on nodes

        f.write("core\n")
        xs = [None] * T
        for i in vehicle:
            xs[i] = rng.integers(CONFIG['vehicle_min_core'],CONFIG['vehicle_max_core']+1)
        for i in edge:
            xs[i] = rng.integers(CONFIG['edge_min_core'],CONFIG['edge_max_core']+1)
        for i in cloud:
            xs[i] = rng.integers(CONFIG['cloud_min_core'],CONFIG['cloud_max_core']+1)
        assert None not in xs
        f.write(f"{' '.join(map(str,xs))}\n")

        f.write("has_camera\n")
        xs = [None] * T
        for i in vehicle:
            xs[i] = 1 if rng.random() < CONFIG['prob_vehicle_needs_camera'] else 0
        for i in edge:
            xs[i] = 1 if rng.random() < CONFIG['prob_edge_needs_camera'] else 0
        for i in cloud: 
            xs[i] = 0
        assert None not in xs
        f.write(f"{' '.join(map(str,xs))}\n")

        f.write("has_gpu\n")
        xs = [None] * T
        for i in vehicle:
            xs[i] = 1 if rng.random() < CONFIG['prob_vehicle_needs_gpu'] else 0
        for i in edge:
            xs[i] = 1 if rng.random() < CONFIG['prob_edge_needs_gpu'] else 0
        for i in cloud: 
            xs[i] = 0
        assert None not in xs
        f.write(f"{' '.join(map(str,xs))}\n")

        # resources/properties on edges

        f.write("bandwidth\n")
        for (u,v) in edges:
            f.write(f"{u},{v} {rng.integers(CONFIG['min_bandwidth'],CONFIG['max_bandwidth']+1)}\n")

        f.write("latency\n")
        for (u,v) in edges:
            f.write(f"{u},{v} {rng.choice(CONFIG['latency'])}\n")

if __name__ == '__main__':

    with open("config_gen_application.json", "r") as f:
        CONFIG = json.load(f)

    assert len(sys.argv) == 5
    CONFIG['app_size'] = int(sys.argv[1])
    CONFIG['app_topology'] = sys.argv[2]
    CONFIG['number_of_topologies_to_generate'] = int(sys.argv[3])
    CONFIG['number_of_property_resources_files_per_topology'] = int(sys.argv[4])

    config_validity_check(CONFIG)

    for seed in range(CONFIG['number_of_topologies_to_generate']):
        generate_application(CONFIG, seed)