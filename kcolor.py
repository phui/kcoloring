import random
import math
import time
import pickle
import ujson as js
import numpy as np
import networkx as nx


from collections import defaultdict
from itertools import izip
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from joblib import Parallel, delayed


import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


ALPHA = 0.001
CLUSTER_PERFORMANCE_METRIC = adjusted_rand_score
NUM_GRAPHS_PER_SETTING = 1#30
NUM_MATCHES_PER_GRAPH = 2#30
NUM_PROC = 8


def random_vector():
    vec = np.random.rand(2) * 2 - 1
    vec = vec / np.linalg.norm(vec)
    return vec

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def repel_grad(node_vec, lb1, lb2):
    vec1, vec2 = node_vec[lb1], node_vec[lb2]
    dot_product = np.dot(vec1, vec2)
    return ALPHA * sigmoid(dot_product) * sigmoid(-dot_product) * vec2

def attract_grad(node_vec, lb1, lb2):
    vec1, vec2 = node_vec[lb1], node_vec[lb2]
    dot_product = np.dot(vec1, vec2)
    return ALPHA * sigmoid(dot_product) * (sigmoid(dot_product) - 1) * vec2

def random_sample(pool, exclude1, exclude2, n):
    selected = set(random.sample(pool, n))
    while exclude1 in selected or exclude2 in selected:
        selected = random.sample(pool, n)
    return selected

def generate_graph(k_part, num_nodes, density, noise_level):
    '''
        assume max entropy i.e. flat distribution of nodes into partitions
    '''
    part_num_nodes = int(1.0 * num_nodes / k_part)
    remainder = num_nodes - part_num_nodes * (k_part - 1)

    parts = tuple([part_num_nodes] * (k_part - 1) + [remainder])
    G = nx.complete_multipartite_graph(*parts)
    complement_edges = set(nx.complement(G).edges())

    # subsample edges to delete in order to achieve the target density
    edges = G.edges()
    deleted_edges = random.sample(edges, int(len(edges)*(1.0-density)))
    G.remove_edges_from(list(deleted_edges))
    deleted_nodes = nx.isolates(G)
    G.remove_nodes_from(list(deleted_nodes))

    num_noise_edges = int(len(G.edges()) * noise_level)
    G.add_edges_from(random.sample(complement_edges, num_noise_edges))

    return G

def check_solution(G, solution):
    true_total, noise_total = 0, 0
    true_correct, noise_correct = 0.0, 0.0

    for node1, node2 in G.edges():
        if G.node[node1]['subset'] == G.node[node2]['subset']:
            noise_total += 1
            if solution[node1] == solution[node2]:
                noise_correct += 1
        else:
            true_total += 1
            if solution[node1] != solution[node2]:
                true_correct += 1

    noise_admission_rate = None
    if noise_total > 0:
        # avoid division by zero
        noise_admission_rate = noise_correct/noise_total

    true_admission_rate = None
    if true_total > 0:
        # avoid division by zero
        true_admission_rate = true_correct/true_total

    return true_admission_rate, noise_admission_rate

def kcolor_match(G, k):
    node_vec = defaultdict(random_vector)
    #num_random_sample = int(math.log(len(G))) + 1
    #num_random_sample = 1

    start = time.clock()
    for _e in range(10000):
        for node in G.nodes():
            for node1, node2 in G.edges(node):
                node_vec[node1] -= repel_grad(node_vec, node1, node2)
                node_vec[node1] /= np.linalg.norm(node_vec[node1])

                #positive_samples = random_sample(nodes, node1, node2, num_random_sample)
                #for sample in positive_samples:
                #    node_vec[node1] -= attract_grad(node_vec, node1, sample)
                #    node_vec[node1] /= np.linalg.norm(node_vec[node1])
    prediction_labels = KMeans(n_clusters=k).fit(np.array(node_vec.values())).labels_
    end = time.clock()

    time_taken = end - start

    ground_truth = [G.node[n]['subset'] for n in node_vec if n in G.node]

    true_admission_rate, noise_admission_rate = check_solution(G,
            {k: lbl for k, lbl in izip(node_vec.keys(),prediction_labels)})

    output = {
        'time_taken': time_taken,
        'true_admission_rate': true_admission_rate,
        'noise_admission_rate': noise_admission_rate,
        'clustering_perf': CLUSTER_PERFORMANCE_METRIC(prediction_labels, ground_truth)
    }
    return output

def experiment(k_part, num_nodes, density, noise_level):
    data = {
        'k': k_part,
        'num_nodes': num_nodes,
        'density' : density,
        'noise_level': noise_level
    }

    performances = list()
    for i_ in xrange(NUM_GRAPHS_PER_SETTING):
        G= generate_graph(
            k_part=k_part,
            num_nodes=num_nodes,
            density=density,
            noise_level=noise_level
        )

        performances.append([
            kcolor_match(G, k_part)
            for j_ in xrange(NUM_MATCHES_PER_GRAPH)
        ])
    data['performances'] = performances
    return js.dumps(data)


def main():

    trial_k_parts = [2, 3, 4, 5]
    trial_num_nodes = [17, 33, 65, 129, 257, 313]
    trial_densities = [1.0, 0.8, 0.6, 0.4, 0.2]
    trial_noise_levels = [0.0, 0.05, 0.1, 0.2]

    '''
    trial_k_parts = [2]
    trial_num_nodes = [17]
    trial_densities = [1.0]
    trial_noise_levels = [0.0, 0.05]
    for k_part in trial_k_parts:
        for num_nodes in trial_num_nodes:
            for density in trial_densities:
                for noise_level in trial_noise_levels:
                    print experiment(k_part, num_nodes, density, noise_level)
    '''
    results = Parallel(n_jobs=NUM_PROC)(
        delayed(experiment)(k_part, num_nodes, density, noise_level)
        for k_part in trial_k_parts
        for num_nodes in trial_num_nodes
        for density in trial_densities
        for noise_level in trial_noise_levels
    )

    with open('results.pickle', 'wb') as f:
        pickle.dump(results, f)


if __name__ == '__main__': main()
