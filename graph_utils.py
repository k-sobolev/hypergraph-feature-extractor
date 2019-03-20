import networkx as nx
import numpy as np
import pandas as pd
from simplcompl import BuildAdjacencyForEdgeComplex, BuildAdjacencyForConventionalTriangleComplex

def dist_abs(x):
    return 1 - np.abs(x)

def get_degrees(G):
    degrees = G.degree()
    degrees_list = list(dict(degrees).values())
    name = f'Degrees_{G.name}'

    return {
        name: degrees_list,
    }

def get_betweenness(G):
    betweenness = nx.betweenness_centrality(G)
    betweenness_list = list(dict(betweenness).values())
    name = f'Betweenness_{G.name}'

    return {
        name: betweenness_list,
    }

def get_closeness(G):
    closeness = nx.closeness_centrality(G)
    closeness_list = list(dict(closeness).values())
    name = f'Closeness_{G.name}'

    return {
        name: closeness_list,
    }

def get_clustering(G):
    clustering = nx.clustering(G)
    clustering_list = list(dict(clustering).values())
    name = f'Clustering_{G.name}'

    return {
        name: clustering_list,
    }

def get_avg_node_degree(G):
    avg_node_degree = nx.average_neighbor_degree(G)
    avg_node_degree_list = list(dict(avg_node_degree).values())
    name = f'AvgNodeDegree_{G.name}'

    return {
        name: avg_node_degree_list,
    }

def get_global_characteristics(G):
    global_dict = {}
    suffix = G.name
    global_dict[f'GlobalEfficiency_{suffix}'] = nx.global_efficiency(G)
    global_dict[f'LocalEfficiency_{suffix}'] = nx.local_efficiency(G)

    return global_dict

def get_graph_features(A, border_value=0.3, min_graph_size=3, max_graph_size=700, max_number_of_edges=400):
    dist_A = dist_abs(A)
    dist_A_binarized = np.zeros_like(dist_A)
    dist_A_binarized[dist_A < border_value] = 1
    dist_A_binarized[np.diag_indices(len(dist_A_binarized))] = 0

    n_edges = dist_A_binarized.sum() / 2
    if n_edges > max_number_of_edges:
        print(f'n_edges is {n_edges}, ignored')
        return None

    print(f'n_edges={n_edges}')
    matrix_triag_0, matrix_triag_1, matrix_triag_2 = BuildAdjacencyForConventionalTriangleComplex(dist_A_binarized)
    matrix_edge_0, matrix_edge_1 = BuildAdjacencyForEdgeComplex(dist_A_binarized)
    n_3click_1simplex, n_3click_2simplex, n_2click_1simplex = len(matrix_triag_1), len(matrix_triag_2), len(matrix_edge_1)
    print(n_3click_1simplex, n_3click_2simplex, n_2click_1simplex)

    max_size = max([n_3click_1simplex, n_3click_2simplex, n_2click_1simplex])
    if max_size > max_graph_size:
        print(f'max_size is {max_size}, ignored')
        return None

    G_initial = nx.convert_matrix.from_numpy_array(dist_A_binarized)
    G_2click_1simplex = nx.convert_matrix.from_numpy_array(matrix_edge_1)
    G_3click_1simplex = nx.convert_matrix.from_numpy_array(matrix_triag_1)
    G_3click_2simplex = nx.convert_matrix.from_numpy_array(matrix_triag_2)

    G_initial.name = 'initial'
    G_2click_1simplex.name = 'G_2click_1simplex'
    G_3click_1simplex.name = 'G_3click_1simplex'
    G_3click_2simplex.name = 'G_3click_2simplex'

    features_dict = {
        'n_edges': n_edges,
        'n_2click_1simplex': n_2click_1simplex,
        'n_3click_1simplex': n_3click_1simplex,
        'n_3click_2simplex': n_3click_2simplex,
    }

    graphs_list = [G_initial, G_2click_1simplex, G_3click_1simplex, G_3click_2simplex]
    methods_list = [get_degrees, get_betweenness, get_closeness, get_clustering, get_avg_node_degree, get_global_characteristics]

    for graph in graphs_list:
        if graph.number_of_nodes() < min_graph_size:
            continue

        for method in methods_list:
            method_features_dict = method(graph)
            features_dict.update(method_features_dict)

    return pd.Series(features_dict)





