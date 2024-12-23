import math
import random
from matplotlib import pyplot as plt
import numpy as np
from algorithms.Algorithm import Algorithm
from algorithms.Graph import Component
from algorithms.KMeansAlgorithm import KMeansAlgorithm
from algorithms.SingleLinkAgglomerativeClusteringAlgorithm import SingleLinkAgglomerativeClusteringAlgorithm
from utils.RunningUtils import get_runner_string
from utils.datasets import generate_points
from utils.GraphGenerator import GraphGenerator
from utils.GraphVisualizer import GraphVisualizer
from algorithms.BoruvkasAlgorithm import BoruvkasAlgorithm

def boruvkas(clean, name, points, n_clusters, ax):
    if clean:
        # clean
        print(f'\n==== BORUVKAS ALGORITHM (CLEAN DATA) ====\n')
        G = GraphGenerator(points.copy()) \
            .to_graph(gen_pair_wise=True)
        gv = GraphVisualizer(G, True, dir = get_runner_string(name, clean, 'runner'))

        def plot(i):
            if len(gv.G.V) >= n_clusters:
                gv.plot_neighborhoods(f'{name} boruvkas round {i}', ax=ax)

        max_iterations = 30
        alg = BoruvkasAlgorithm(G, max_iterations, False, plot)
        alg.run()
    else:
        print(f'\n==== BORUVKAS ALGORITHM (NOISY) ====\n')
        # with noise
        G = get_noise(points)
        gv = GraphVisualizer(G, True, dir = get_runner_string(name, clean, 'runner'))

        def plot(i):
            if len(gv.G.V) >= n_clusters:
                gv.plot_neighborhoods(f'{name} boruvkas round {i} (noise)', ax=ax)

        max_iterations = 30
        alg = BoruvkasAlgorithm(G, max_iterations, False, plot)
        _ = alg.run()


def kmeans(clean, name, points, n_clusters, ax):
    voronoi = n_clusters >= 3
    if clean:
        # clean
        print(f'\n==== KMEANS ALGORITHM (CLEAN DATA) ====\n')

        G = GraphGenerator(points.copy()) \
            .to_graph(gen_pair_wise=True)
        gv = GraphVisualizer(G, True, dir = get_runner_string(name, clean, 'runner'))

        def plot(i, centroids):
            gv.plot_kmeans(f'{name} kmeans round {i}', centroids=centroids, voronoi=voronoi, ax=ax)

        max_iterations = 300
        alg = KMeansAlgorithm(G, max_iterations, n_clusters, False, lambda i, centroids: plot(i, centroids))
        alg.run()
    else:
        print(f'\n==== KMEANS ALGORITHM (NOISY) ====\n')

        # with noise
        G = get_noise(points)
        gv = GraphVisualizer(G, True, dir = get_runner_string(name, clean, 'runner'))
        
        def plot(i, centroids):
            gv.plot_kmeans(f'{name} kmeans round {i} (noise)', centroids=centroids, voronoi = voronoi, ax=ax)

        max_iterations = 300
        alg = KMeansAlgorithm(G, max_iterations, n_clusters, False, lambda i, centroids: plot(i, centroids))
        alg.run()

def singlelink(clean, name, points, n_clusters, ax):
    if clean:
        # clean
        print(f'\n==== SLAC ALGORITHM (CLEAN DATA) ====\n')

        G = GraphGenerator(points.copy()) \
            .to_graph(gen_pair_wise=True)
        gv = GraphVisualizer(G, True, dir = get_runner_string(name, clean, 'runner'))

        max_iterations = len(G.V)

        def plot(i):
            if len([component for component in gv.G.components if type(component) is Component]) >= n_clusters:
                gv.plot_component_graph(f'{name} SLAC round {i}', ax=ax)

        alg = SingleLinkAgglomerativeClusteringAlgorithm(G, max_iterations, False, plot)
        start_plotting_at_n_components = 15
        alg.run(start_plotting_at_n_components)
    else:
        print(f'\n==== SLAC ALGORITHM (NOISY) ====\n')
        
        # with noise
        G = get_noise(points)
        gv = GraphVisualizer(G, True, dir = get_runner_string(name, clean, 'runner'))

        max_iterations = len(G.V)

        def plot(i):
            if len([component for component in gv.G.components if type(component) is Component]) >= n_clusters:
                gv.plot_component_graph(f'{name} SLAC round {i} (with noise)', ax=ax)

        alg = SingleLinkAgglomerativeClusteringAlgorithm(G, max_iterations, False, plot)
        start_plotting_at_n_components = 15
        alg.run(start_plotting_at_n_components)

def get_noise(points):
        # .add_point((8, 7)) \
        # .add_point((7, 7)) \
        # .add_clustered_noise('circle', 0.1) \
        # .draw_line((-1, 5), (0, -6.), 1.5) \
    G = GraphGenerator(points.copy()) \
        .add_gaussian_noise(0.1) \
        .to_graph(gen_pair_wise=True)
    return G

def run():
    dataset_names = ['circles'] 
    # dataset_names = ['ans1c'] 
    # dataset_names = ['ans1a', 'ans1b', 'ans1c',] 
    # dataset_names = ['ans1a', 'ans1b', 'ans1c', 'blobs', 'moons', 'circles', 'maccie']
    n_samples = 500
    n_clusters = 2
    for clean in [True, False]: # True
        for dataset in dataset_names:
            fig = plt.figure(figsize=(16, 4))
            axs = fig.subplots(1, 3, sharex=True, sharey=True)
            points = generate_points(dataset, n_samples, n_clusters)
            kmeans(clean, dataset, points, n_clusters, axs[1])
            boruvkas(clean, dataset, points, n_clusters, axs[0])
            singlelink(clean, dataset, points, n_clusters, axs[2])
        

if __name__ == "__main__":
    np.random.seed(56)
    random.seed(56)
    run()