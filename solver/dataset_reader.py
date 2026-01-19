import os

def instances_iterator(
    NETWORK_SIZES,
    NUMBER_OF_TOPOLOGIES_PER_SIZE,
    NUMBER_OF_RP_FILES_PER_TOPOLOGY,
    MIN_NAPPS, 
    MAX_NAPPS,
    INSTANCES_PER_NAPPS
):
    
    for network_size in NETWORK_SIZES:
        for seed in range(NUMBER_OF_TOPOLOGIES_PER_SIZE):
            for rp in range(NUMBER_OF_RP_FILES_PER_TOPOLOGY):
                filename_network_topology = f"n_{network_size}_s_{seed}.dat"
                filename_network_rp = f"n_{network_size}_s_{seed}_rp_{rp}.dat"
                id = 0
                for napps in range(MIN_NAPPS,MAX_NAPPS+1):
                    for _ in range(INSTANCES_PER_NAPPS):
                        filename_app_set = f"id_{id}_{napps}_apps.dat"
                        with open(os.path.join("../dataset/app_sets", filename_app_set), 'r') as f:
                            filenames_app_rp = list(map(lambda line: line.strip(), f.readlines()))
                        filenames_app_topology = list(map(lambda name: f"{name.split('_rp_')[0]}.dat", filenames_app_rp))
                        
                        experiment_id = f"network-{filename_network_rp.rstrip('.dat')}-app_set-{filename_app_set.rstrip('.dat')}"
                        instance = {
                            "experiment_id" : experiment_id,
                            "filename_network_topology" : filename_network_topology,
                            "filename_network_rp" : filename_network_rp,
                            "network_size" : network_size,
                            "network_seed" : seed,
                            "network_rp" : rp,
                            "filename_app_set" : filename_app_set,
                            "id": id,
                            "napps": napps,
                            "filenames_app_rp": filenames_app_rp,
                            "filenames_app_topology": filenames_app_topology,
                        }
                        yield instance
                        id += 1