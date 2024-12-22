from matplotlib import pyplot as plt
from algorithms.Algorithm import Algorithm
from algorithms.Graph import Component
from algorithms.KMeansAlgorithm import KMeansAlgorithm
from algorithms.SingleLinkAgglomerativeClusteringAlgorithm import SingleLinkAgglomerativeClusteringAlgorithm
from utils.datasets import generate_points
from utils.GraphGenerator import GraphGenerator
from utils.GraphVisualizer import GraphVisualizer
from algorithms.BoruvkasAlgorithm import BoruvkasAlgorithm

def get_runner_string(name, clean, dir):
    sub = 'clean'
    if not clean:
        sub = 'noisy'
    return f'{name}/{dir}/{sub}'

def boruvkas(clean, name, points, ax):
    if clean:
        # clean
        print(f'\n==== BORUVKAS ALGORITHM (CLEAN DATA) ====\n')
        G = GraphGenerator(points.copy()) \
            .to_graph(gen_pair_wise=True)
        gv = GraphVisualizer(G, True, dir = get_runner_string(name, clean, 'runner'))

        def plot(i):
            if len(gv.G.V) >= 3:
                gv.plot_graph(f'{name} boruvkas round {i}', ax=ax)

        max_iterations = 30
        alg = BoruvkasAlgorithm(G, max_iterations, False, plot)
        alg.run()
    else:
        print(f'\n==== BORUVKAS ALGORITHM (NOISY) ====\n')
        # with noise
        G = GraphGenerator(points.copy()) \
            .add_gaussian_noise(0.1) \
            .to_graph(gen_pair_wise=True)
        gv = GraphVisualizer(G, True, dir = get_runner_string(name, clean, 'runner'))

        def plot(i):
            if len(gv.G.V) >= 3:
                gv.plot_graph(f'{name} boruvkas round {i} (noise)', ax=ax)

        max_iterations = 30
        alg = BoruvkasAlgorithm(G, max_iterations, False, plot)
        _ = alg.run()


def kmeans(clean, name, points, n_clusters, ax):
    if clean:
        # clean
        print(f'\n==== KMEANS ALGORITHM (CLEAN DATA) ====\n')

        G = GraphGenerator(points.copy()) \
            .to_graph(gen_pair_wise=True)
        gv = GraphVisualizer(G, True, dir = get_runner_string(name, clean, 'runner'))

        def plot(i, centroids):
            gv.plot_kmeans(f'{name} kmeans round {i}', centroids=centroids, voronoi=True, ax=ax)

        max_iterations = 300
        alg = KMeansAlgorithm(G, max_iterations, n_clusters, False, lambda i, centroids: plot(i, centroids))
        alg.run()
    else:
        print(f'\n==== KMEANS ALGORITHM (NOISY) ====\n')

        # with noise
        G = GraphGenerator(points.copy()) \
            .add_gaussian_noise(0.1) \
            .to_graph(gen_pair_wise=True)
        gv = GraphVisualizer(G, True, dir = get_runner_string(name, clean, 'runner'))
        
        def plot(i, centroids):
            gv.plot_kmeans(f'{name} kmeans round {i} (noise)', centroids=centroids, voronoi=True, ax=ax)

        max_iterations = 300
        alg = KMeansAlgorithm(G, max_iterations, n_clusters, False, lambda i, centroids: plot(i, centroids))
        alg.run()

def singlelink(clean, name, points, ax):
    if clean:
        # clean
        print(f'\n==== SLAC ALGORITHM (CLEAN DATA) ====\n')

        G = GraphGenerator(points.copy()) \
            .to_graph(gen_pair_wise=True)
        gv = GraphVisualizer(G, True, dir = get_runner_string(name, clean, 'runner'))

        max_iterations = len(points)

        def plot(i):
            if len([component for component in gv.G.components if type(component) is Component]) >= 3:
                gv.plot_component_graph(f'{name} SLAC round {i}', ax=ax)

        alg = SingleLinkAgglomerativeClusteringAlgorithm(G, max_iterations, False, plot)
        start_plotting_at_n_components = 15
        alg.run(start_plotting_at_n_components)
    else:
        print(f'\n==== SLAC ALGORITHM (NOISY) ====\n')
        
        # with noise
        G = GraphGenerator(points.copy()) \
            .add_gaussian_noise(0.1) \
            .to_graph(gen_pair_wise=True)
        gv = GraphVisualizer(G, True, dir = get_runner_string(name, clean, 'runner'))

        max_iterations = len(points)

        def plot(i):
            if len([component for component in gv.G.components if type(component) is Component]) >= 3:
                gv.plot_component_graph(f'{name} SLAC round {i} (with noise)', ax=ax)

        alg = SingleLinkAgglomerativeClusteringAlgorithm(G, max_iterations, False, plot)
        start_plotting_at_n_components = 15
        alg.run(start_plotting_at_n_components)

def run():
    # dataset_names = ['ans1c'] 
    dataset_names = ['ans1a', 'ans1b', 'ans1c', 'blobs', 'moons', 'circles', 'maccie']
    n_samples = 500
    n_clusters = 3
    for clean in [True, False]:
        for dataset in dataset_names:
            fig = plt.figure(figsize=(16, 4))
            axs = fig.subplots(1, 3, sharex=True, sharey=True)
            points = generate_points(dataset, n_samples, n_clusters)
            kmeans(clean, dataset, points, n_clusters, axs[1])
            boruvkas(clean, dataset, points, axs[0])
            singlelink(clean, dataset, points, axs[2])
        

if __name__ == "__main__":
    run()