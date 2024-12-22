from matplotlib import pyplot as plt
from algorithms.Algorithm import Algorithm
from algorithms.KMeansAlgorithm import KMeansAlgorithm
from algorithms.SingleLinkAgglomerativeClusteringAlgorithm import SingleLinkAgglomerativeClusteringAlgorithm
from utils.datasets import generate_points
from utils.GraphGenerator import GraphGenerator
from utils.GraphVisualizer import GraphVisualizer
from algorithms.BoruvkasAlgorithm import BoruvkasAlgorithm


def boruvkas(name, points):
    # clean
    print(f'\n==== BORUVKAS ALGORITHM (CLEAN DATA) ====\n')
    G = GraphGenerator(points.copy()) \
        .to_graph(gen_pair_wise=True)
    
    gv = GraphVisualizer(G, True)
    gv.plot_graph(f'{name}')

    max_iterations = 10
    plot_graph = lambda i: gv.plot_graph(f'{name} boruvkas round {i}', color=True)
    alg = BoruvkasAlgorithm(G, max_iterations, False, plot_graph)
    alg.run()
    
    print(f'\n==== BORUVKAS ALGORITHM (NOISY) ====\n')
    # with noise
    G = GraphGenerator(points.copy()) \
        .add_gaussian_noise(0.1) \
        .to_graph(gen_pair_wise=True)

    gv = GraphVisualizer(G, True)
    gv.plot_graph(f'{name} with gauss noise')

    max_iterations = 10
    plot_graph = lambda i: gv.plot_graph(f'{name} boruvkas round {i} (noise)', color=True)
    alg = BoruvkasAlgorithm(G, max_iterations, False, plot_graph)
    _ = alg.run()


def kmeans(name, points, n_clusters):
    # clean
    print(f'\n==== KMEANS ALGORITHM (CLEAN DATA) ====\n')

    G = GraphGenerator(points.copy()) \
        .to_graph(gen_pair_wise=True)
    
    gv = GraphVisualizer(G, True)
    gv.plot_graph(f'{name}')

    max_iterations = 300
    plot_graph = lambda i, centroids: gv.plot_kmeans(f'{name} kmeans round {i}', centroids=centroids, voronoi=True)
    alg = KMeansAlgorithm(G, max_iterations, n_clusters, False, plot_graph)
    alg.run()
    
    print(f'\n==== KMEANS ALGORITHM (NOISY) ====\n')

    # with noise
    G = GraphGenerator(points.copy()) \
        .add_gaussian_noise(0.1) \
        .to_graph(gen_pair_wise=True)

    gv = GraphVisualizer(G, True)
    gv.plot_graph(f'{name} with gauss noise')

    max_iterations = 10
    plot_graph = lambda i, centroids: gv.plot_kmeans(f'{name} kmeans round {i} (noise)', centroids=centroids, voronoi=True)
    alg = KMeansAlgorithm(G, max_iterations, n_clusters, False, plot_graph)
    alg.run()

def singlelink(name, points):
    # clean
    print(f'\n==== SLAC ALGORITHM (CLEAN DATA) ====\n')

    G = GraphGenerator(points.copy()) \
        .to_graph(gen_pair_wise=True)
    
    gv = GraphVisualizer(G, True)
    gv.plot_graph(f'{name}')

    max_iterations = len(points)
    plot_graph = lambda i: gv.plot_component_graph(f'{name} SLAC round {i}')
    alg = SingleLinkAgglomerativeClusteringAlgorithm(G, max_iterations, False, plot_graph)
    alg.run()

    print(f'\n==== SLAC ALGORITHM (NOISY) ====\n')
    
    # with noise
    G = GraphGenerator(points.copy()) \
        .add_gaussian_noise(0.1) \
        .to_graph(gen_pair_wise=True)

    gv = GraphVisualizer(G, True)
    gv.plot_graph(f'{name} with gauss noise')

    max_iterations = len(points)
    plot_graph = lambda i: gv.plot_component_graph(f'{name} SLAC round {i} (with noise)')
    alg = SingleLinkAgglomerativeClusteringAlgorithm(G, max_iterations, False, plot_graph)
    alg.run()

def run():
    dataset_names = ['ans1c'] 
    # dataset_names = ['ans1a', 'ans1b', 'ans1c', 'blobs', 'moons', 'circles', 'maccie']
    n_samples = 500
    n_clusters = 3
    for dataset in dataset_names:
        points = generate_points(dataset, n_samples, n_clusters)
        boruvkas(dataset, points)
        kmeans(dataset, points, n_clusters)
        singlelink(dataset, points)
        

if __name__ == "__main__":
    run()