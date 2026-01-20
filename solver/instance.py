from dataclasses import dataclass
from typing import Any
import os
import numpy as np

from instance_utils import (
    build_network_structure_from_file,
    build_network_rp_availability_from_file,
    build_app_structure_from_file,
    build_app_rp_consumption_from_file,
    build_b_coefficients
)

@dataclass(frozen=True)
class App:

    T: Any
    D: Any

    q_microservices_R_core: Any
    q_microservices_S_has_camera: Any
    q_microservices_S_has_gpu: Any
    q_connections_R_bandwidth: Any
    q_connections_S_latency: Any

    b_microservices_zero: Any
    b_microservices_one: Any
    b_connections_zero_not_implied: Any
    b_connections_one: Any
    b_connections_one_actual: Any

@dataclass(frozen=True)
class Instance:

    R: Any
    S: Any
    K: Any

    I: Any
    A: Any
    c: Any
    P: Any
    a2: Any

    Q_nodes_R_core: Any
    Q_nodes_S_has_camera: Any 
    Q_nodes_S_has_gpu: Any
    Q_links_R_bandwidth: Any
    Q_links_S_latency: Any

    apps: Any
    app_merged: Any

def build_merged_app(apps):
    
    merged_T = sum(a.T for a in apps)
    merged_D = set()
    merged_q_microservices_R_core = np.full(merged_T, -1, dtype=np.int32)
    merged_q_microservices_S_has_camera = np.full(merged_T, -1, dtype=np.int32)
    merged_q_microservices_S_has_gpu = np.full(merged_T, -1, dtype=np.int32)
    merged_q_connections_R_bandwidth = np.full((merged_T, merged_T), -1, dtype=np.int32)
    merged_q_connections_S_latency = np.full((merged_T, merged_T), -1, dtype=np.int32)

    shift = 0
    idx = 0
    for a in apps:

        assert idx == shift
        for x1,x2,x3 in zip(
            a.q_microservices_R_core,
            a.q_microservices_S_has_camera,
            a.q_microservices_S_has_gpu):
            merged_q_microservices_R_core[idx] = x1
            merged_q_microservices_S_has_camera[idx] = x2
            merged_q_microservices_S_has_gpu[idx] = x3
            idx += 1
        
        for (u,v) in a.D:
            merged_D.add((u+shift,v+shift))
            merged_q_connections_R_bandwidth[u+shift,v+shift] = a.q_connections_R_bandwidth[u,v]
            merged_q_connections_S_latency[u+shift,v+shift] = a.q_connections_S_latency[u,v] 

        shift += a.T

    assert merged_T == shift
    assert merged_T == idx
    assert -1 not in merged_q_microservices_R_core
    assert -1 not in merged_q_microservices_S_has_camera
    assert -1 not in merged_q_microservices_S_has_gpu

    return (
        merged_T, merged_D,
        merged_q_microservices_R_core,
        merged_q_microservices_S_has_camera,
        merged_q_microservices_S_has_gpu,
        merged_q_connections_R_bandwidth,
        merged_q_connections_S_latency
    )


def create_instance(DATASET_NETWORK_DIR, DATASET_APP_DIR, ins):

    R = {'bandwidth', 'core'}
    S = {'has_gpu', 'latency', 'has_camera'}
    K = {'bandwidth', 'core', 'has_gpu', 'latency', 'has_camera'}

    I,A,c,P,a2 = build_network_structure_from_file(
        os.path.join(DATASET_NETWORK_DIR, ins['filename_network_topology'])
    )

    (
        Q_nodes_R_core, Q_nodes_S_has_camera, Q_nodes_S_has_gpu, 
        Q_links_R_bandwidth, Q_links_S_latency 
    ) = build_network_rp_availability_from_file(
        os.path.join(DATASET_NETWORK_DIR, ins['filename_network_rp']), 
        I, A
    )
    
    apps = []
    assert ins['napps'] == len(ins['filenames_app_topology']) == len(ins['filenames_app_rp'])
    for i in range(ins['napps']):

        T,D = build_app_structure_from_file(
            os.path.join(DATASET_APP_DIR, ins['filenames_app_topology'][i])
        ) 

        (
            q_microservices_R_core, q_microservices_S_has_camera, q_microservices_S_has_gpu, 
            q_connections_R_bandwidth, q_connections_S_latency
        ) = build_app_rp_consumption_from_file(
            os.path.join(DATASET_APP_DIR, ins['filenames_app_rp'][i]), 
            T,D
        )

        (
            b_microservices_zero, b_microservices_one,
            b_connections_zero_not_implied, b_connections_one, b_connections_one_actual
        ) = build_b_coefficients(
            I, P,
            Q_nodes_R_core, Q_links_R_bandwidth, 
            Q_nodes_S_has_camera, Q_nodes_S_has_gpu, Q_links_S_latency,
            T,D,
            q_microservices_R_core, q_microservices_S_has_camera, q_microservices_S_has_gpu,
            q_connections_R_bandwidth, q_connections_S_latency
        ) # can raise InfeasibleInstance exception

        apps.append(
            App(
                T,D,
                q_microservices_R_core, q_microservices_S_has_camera, q_microservices_S_has_gpu,
                q_connections_R_bandwidth, q_connections_S_latency,
                b_microservices_zero, b_microservices_one,
                b_connections_zero_not_implied, b_connections_one, b_connections_one_actual
            )
        )
    
    (
        T, D,
        q_microservices_R_core, q_microservices_S_has_camera, q_microservices_S_has_gpu,
        q_connections_R_bandwidth, q_connections_S_latency
    ) = build_merged_app(apps)

    (
        b_microservices_zero, b_microservices_one,
        b_connections_zero_not_implied, b_connections_one, b_connections_one_actual
    ) = build_b_coefficients(
        I, P,
        Q_nodes_R_core, Q_links_R_bandwidth, 
        Q_nodes_S_has_camera, Q_nodes_S_has_gpu, Q_links_S_latency,
        T,D,
        q_microservices_R_core, q_microservices_S_has_camera, q_microservices_S_has_gpu,
        q_connections_R_bandwidth, q_connections_S_latency
    ) # can raise InfeasibleInstance exception

    app_merged = App(
        T,D,
        q_microservices_R_core, q_microservices_S_has_camera, q_microservices_S_has_gpu,
        q_connections_R_bandwidth, q_connections_S_latency,
        b_microservices_zero, b_microservices_one,
        b_connections_zero_not_implied, b_connections_one, b_connections_one_actual
    )

    shift = 0
    for a in apps:
        for u in range(a.T):
            assert a.b_microservices_zero[u] == app_merged.b_microservices_zero[u+shift]
            assert a.b_microservices_one[u] == app_merged.b_microservices_one[u+shift]
        for u in range(a.T):
            for v in range(a.T):
                assert a.b_connections_zero_not_implied[u][v] == app_merged.b_connections_zero_not_implied[u+shift][v+shift]
                assert a.b_connections_one[u][v] == app_merged.b_connections_one[u+shift][v+shift]
                assert a.b_connections_one_actual[u][v] == app_merged.b_connections_one_actual[u+shift][v+shift]
        shift += a.T

    return Instance(
        R,S,K,
        I,A,c,P,a2,
        Q_nodes_R_core, Q_nodes_S_has_camera, Q_nodes_S_has_gpu,
        Q_links_R_bandwidth, Q_links_S_latency,
        apps,
        app_merged
    )