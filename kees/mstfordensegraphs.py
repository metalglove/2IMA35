import math
from argparse import ArgumentParser
from datetime import datetime

import numpy as np
import random
import matplotlib.pyplot as plt
import scipy.spatial

from sklearn.datasets import make_circles, make_moons, make_blobs

# Snap stanford

def get_clustering_data():
    n_samples = 1500
    noisy_circles = make_circles(n_samples=n_samples, factor=.5,
                                          noise=.05)
    noisy_moons = make_moons(n_samples=n_samples, noise=.05)
    blobs = make_blobs(n_samples=n_samples, random_state=8)
    no_structure = np.random.rand(n_samples, 2), None

    # Anisotropicly distributed data
    random_state = 170
    X, y = make_blobs(n_samples=n_samples, random_state=random_state)
    transformation = [[0.6, -0.6], [-0.4, 0.8]]
    X_aniso = np.dot(X, transformation)
    aniso = (X_aniso, y)

    # blobs with varied variances
    varied = make_blobs(n_samples=n_samples,
                                 cluster_std=[1.0, 2.5, 0.5],
                                 random_state=random_state)

    plt.figure(figsize=(9 * 2 + 3, 13))
    plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.95, wspace=.05,
                        hspace=.01)

    plot_num = 1

    default_base = {'quantile': .3,
                    'eps': .3,
                    'damping': .9,
                    'preference': -200,
                    'n_neighbors': 10,
                    'n_clusters': 3,
                    'min_samples': 20,
                    'xi': 0.05,
                    'min_cluster_size': 0.1}

    datasets = [
        (noisy_circles, {'damping': .77, 'preference': -240,
                         'quantile': .2, 'n_clusters': 2,
                         'min_samples': 20, 'xi': 0.25}),
        (noisy_moons, {'damping': .75, 'preference': -220, 'n_clusters': 2}),
        (varied, {'eps': .18, 'n_neighbors': 2,
                  'min_samples': 5, 'xi': 0.035, 'min_cluster_size': .2}),
        (aniso, {'eps': .15, 'n_neighbors': 2,
                 'min_samples': 20, 'xi': 0.1, 'min_cluster_size': .2}),
        (blobs, {}),
        (no_structure, {})]

    return datasets


def create_distance_matrix(dataset):
    x = []
    y = []
    size = 0
    for line in dataset:
        x.append([line[0]])
        y.append([line[1]])
    d_matrix = scipy.spatial.distance_matrix(x, y, threshold=1000000)
    dict = {}
    for i in range(len(d_matrix)):
        dict2 = {}
        for j in range(len(d_matrix[i])):
            if i != j:
                size += 1
                dict2[j] = d_matrix[i][j]
        dict[i] = dict2
    return d_matrix, dict, size


def partion_vertices(vertices, k):
    U = []
    V = []
    random.shuffle(vertices)
    verticesU = vertices.copy()
    random.shuffle(vertices)
    verticesV = vertices.copy()
    for i in range(len(vertices)):
        if i < k:
            U.append({verticesU[i]})
            V.append({verticesV[i]})
        else:
            U[i % k].add(verticesU[i])
            V[i % k].add(verticesV[i])
    return U, V


def get_key(item):
    return item[2]


def find_mst(V, U, E):
    e_copy = E.copy()
    vertices = set()
    for v in V:
        vertices.add(v)
    for u in U:
        vertices.add(u)
    e_copy = sorted(e_copy, key=get_key)
    connected_component = set()
    mst = set()
    remove_edges = set()
    while len(mst) < len(vertices) - 1:
        for edge in e_copy:
            if len(connected_component) == 0:
                connected_component.add(edge[0])
                connected_component.add(edge[1])
                mst.add(edge)
                E.remove(edge)
                break
            else:
                if edge[0] in connected_component:
                    if edge[1] in connected_component:
                        remove_edges.add(edge)
                        e_copy.remove(edge)
                    else:
                        connected_component.add(edge[1])
                        mst.add(edge)
                        e_copy.remove(edge)
                        break
                elif edge[1] in connected_component:
                    if edge[0] in connected_component:
                        remove_edges.add(edge)
                        e_copy.remove(edge)
                    else:
                        connected_component.add(edge[0])
                        mst.add(edge)
                        e_copy.remove(edge)
                        break
    for edge in e_copy:
        remove_edges.add(edge)
    return mst, remove_edges


"""
get_edges that works without the pyspark implementation

Here U and V are the partitionings of the graph and for every combination it returns the edges corresponding to this graph
"""
def get_edges(U, V, E):
    if len(U) == len(V) == 1:
        return E.items()
    subgraphs = []
    for u in U:
        first = []
        for v in V:
            edges = []
            for node1 in u:
                for node2 in v:
                    if node2 in E[node1]:
                        edges.append((node1, node2, E[node1][node2]))
            first.append(edges)
        subgraphs.append(first)
    return subgraphs


def reduce_edges(vertices, E, c, epsilon):
    n = len(vertices)
    k = math.ceil(n**((c - epsilon) / 2))
    U, V = partion_vertices(vertices, k)
    subgraphs = get_edges(U, V, E)
    removed = set()
    mst = set()
    for i in range(len(U)):
        for j in range(len(V)):
            mst, removed_edges = find_mst(U[i], V[j], subgraphs[i][j])
            removed = removed.union(removed_edges)
    return mst, removed


"""
Input: E = current edges, removed_edges = edges to be removed from the edges, mst = edges that should not be removed
Output: E = updated edges where removed_edges are not part of it
"""
def remove_edges(E, removed_edges, mst):
    for edge in removed_edges:
        if edge[1] in E[edge[0]]:
            del E[edge[0]][edge[1]]
        if edge[0] in E[edge[1]]:
            del E[edge[1]][edge[0]]
    for edge in mst:
        if edge[1] not in E[edge[0]]:
            E[edge[0]][edge[1]] = edge[2]
        if edge[0] not in E[edge[1]]:
            E[edge[1]][edge[0]] = edge[2]
    return E


def create_mst(V, E, epsilon, m, size):
    n = len(V)
    c = math.log(m / n, n)
    while size > np.power(n, 1 + epsilon):
        mst, removed_edges = reduce_edges(V, E, c, epsilon)
        E = remove_edges(E, removed_edges, mst)
        size = size - len(removed_edges)
        c = (c - epsilon) / 2
    return E


def main(machines, epsilon):
    parser = ArgumentParser()
    parser.add_argument('--test', help="Used for smaller dataset and testing", action="store_true")
    args = parser.parse_args()

    print("Start generating MST")
    if args.test:
        print("Test argument given")

    start_time = datetime.now()
    print("Starting time:", start_time)

    # create_mst()
    datasets = get_clustering_data()

    for dataset in datasets:
        timestamp = datetime.now()
        print("Start creating Distance Matrix...")
        dm, E, size = create_distance_matrix(dataset[0][0])
        V = list(range(len(dm)))
        print("Created distance matrix in: ", datetime.now() - timestamp)
        print("Start creating MST...")
        timestamp = datetime.now()
        E = create_mst(V, E, epsilon=epsilon, m=machines, size=size)
        U, V = set(range(1500))
        E = get_edges(U, V, E)
        mst, removed_edges = find_mst(U[0], V[0], E[0][0])
        print("Created MST in: ", datetime.now() - timestamp)
        # print("MST:\n", mst)
        print("Start creating plot of MST...")
        timestamp = datetime.now()
        print("TODO...")
        print("Created plot of MST in: ", datetime.now() - timestamp)
        break


if __name__ == '__main__':
    machines = 4
    c = 1/2 # 0 <= c <= 1
    epsilon = 1/8
    main(machines=machines, epsilon=epsilon)
