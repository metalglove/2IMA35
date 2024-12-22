from matplotlib import pyplot as plt
from algorithms.Algorithm import Algorithm
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
    plotted = False
    if clean:
        # clean
        print(f'\n==== BORUVKAS ALGORITHM (CLEAN DATA) ====\n')
        G = GraphGenerator(points.copy()) \
            .to_graph(gen_pair_wise=True)
        gv = GraphVisualizer(G, True, dir = get_runner_string(name, clean, 'runner'))

        def plot(i):
            gv.plot_graph(f'{name} boruvkas round {i}', ax=ax)

        max_iterations = 10
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
            gv.plot_graph(f'{name} boruvkas round {i} (noise)', ax=ax)

        max_iterations = 10
        alg = BoruvkasAlgorithm(G, max_iterations, False, plot)
        _ = alg.run()


def kmeans(clean, name, points, n_clusters, ax):
    plotted = False
    if clean:
        # clean
        print(f'\n==== KMEANS ALGORITHM (CLEAN DATA) ====\n')

        G = GraphGenerator(points.copy()) \
            .to_graph(gen_pair_wise=True)
        gv = GraphVisualizer(G, True, dir = get_runner_string(name, clean, 'runner'))

        def plot(i, centroids, plotted):
            gv.plot_kmeans(f'{name} kmeans round {i}', centroids=centroids, voronoi=True, ax=ax)
            plotted = True

        max_iterations = 300
        alg = KMeansAlgorithm(G, max_iterations, n_clusters, False, lambda i, centroids: plot(i, centroids, plotted))
        alg.run()
    else:
        print(f'\n==== KMEANS ALGORITHM (NOISY) ====\n')

        # with noise
        G = GraphGenerator(points.copy()) \
            .add_gaussian_noise(0.1) \
            .to_graph(gen_pair_wise=True)
        gv = GraphVisualizer(G, True, dir = get_runner_string(name, clean, 'runner'))
        
        def plot(i, centroids, plotted):
            gv.plot_kmeans(f'{name} kmeans round {i} (noise)', centroids=centroids, voronoi=True, ax=ax)
            plotted = True

        max_iterations = 10
        alg = KMeansAlgorithm(G, max_iterations, n_clusters, False, lambda i, centroids: plot(i, centroids, plotted))
        alg.run()

def singlelink(clean, name, points, ax):
    plotted = False

    if clean:
        # clean
        print(f'\n==== SLAC ALGORITHM (CLEAN DATA) ====\n')

        G = GraphGenerator(points.copy()) \
            .to_graph(gen_pair_wise=True)
        gv = GraphVisualizer(G, True, dir = get_runner_string(name, clean, 'runner'))

        max_iterations = len(points)

        def plot(i, plotted):
            gv.plot_component_graph(f'{name} SLAC round {i}', ax=ax)
            plotted = True

        alg = SingleLinkAgglomerativeClusteringAlgorithm(G, max_iterations, False, lambda i: plot(i, plotted))
        alg.run()
    else:
        print(f'\n==== SLAC ALGORITHM (NOISY) ====\n')
        
        # with noise
        G = GraphGenerator(points.copy()) \
            .add_gaussian_noise(0.1) \
            .to_graph(gen_pair_wise=True)
        gv = GraphVisualizer(G, True, dir = get_runner_string(name, clean, 'runner'))

        max_iterations = len(points)

        def plot(i, plotted):
            gv.plot_component_graph(f'{name} SLAC round {i} (with noise)', ax=ax)
            plotted = True

        alg = SingleLinkAgglomerativeClusteringAlgorithm(G, max_iterations, False, lambda i: plot(i, plotted))
        start_plotting_at_n_components = 10
        alg.run(start_plotting_at_n_components)

def run():
    dataset_names = ['ans1c'] 
    # dataset_names = ['ans1a', 'ans1b', 'ans1c', 'blobs', 'moons', 'circles', 'maccie']
    n_samples = 500
    n_clusters = 3
    for clean in [True, False]:
        for dataset in dataset_names:
            fig = plt.figure(figsize=(16, 4))
            axs = fig.subplots(1, 3, sharex=True, sharey=True)
            points = generate_points(dataset, n_samples, n_clusters)
            singlelink(clean, dataset, points, axs[2])
            kmeans(clean, dataset, points, n_clusters, axs[1])
            boruvkas(clean, dataset, points, axs[0])
        

if __name__ == "__main__":
    run()