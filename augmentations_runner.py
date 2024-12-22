import random
from matplotlib import pyplot as plt
import numpy as np
from algorithms.Algorithm import Algorithm
from algorithms.KMeansAlgorithm import KMeansAlgorithm
from algorithms.SingleLinkAgglomerativeClusteringAlgorithm import SingleLinkAgglomerativeClusteringAlgorithm
from utils.datasets import generate_points
from utils.GraphGenerator import GraphGenerator
from utils.GraphVisualizer import GraphVisualizer
from algorithms.BoruvkasAlgorithm import BoruvkasAlgorithm

def get_runner_string(name,  dir):
    return f'{name}/{dir}'

def gauss(dataset, points, noise, ax):
    G = GraphGenerator(points.copy()) \
        .add_gaussian_noise(0.1) \
        .to_graph(gen_pair_wise=True)

    gv = GraphVisualizer(G, True, dir = get_runner_string(dataset, 'augmentations_runner/gauss'))
    gv.plot_graph(f'{dataset} with gauss noise={noise}', ax = ax)

def draw_line(dataset, points, start_point, end_point, spacing, ax):
    G = GraphGenerator(points.copy()) \
        .draw_line(start_point, end_point, spacing) \
        .to_graph(gen_pair_wise=True)

    gv = GraphVisualizer(G, True, dir = get_runner_string(dataset, 'augmentations_runner/line'))
    gv.plot_graph(f'{dataset} with a line from ({start_point}) to ({end_point}) spacing={spacing}', ax = ax)

def clustered_noise(dataset, points, type, noise, ax):
    G = GraphGenerator(points.copy()) \
        .add_clustered_noise(type, noise) \
        .to_graph(gen_pair_wise=True)

    gv = GraphVisualizer(G, True, dir = get_runner_string(dataset, f'augmentations_runner/{type}'))
    gv.plot_graph(f'{dataset} with type={type} noise={noise}', ax = ax)

def run():
    dataset_names = ['blobs'] 
    # dataset_names = ['ans1a', 'ans1b', 'ans1c', 'blobs', 'moons', 'circles', 'maccie']
    n_samples = 500
    n_clusters = 3
    for dataset in dataset_names:
        fig = plt.figure(figsize=(16, 4))
        axs = fig.subplots(1, 3, sharex=True, sharey=True)
        points = generate_points(dataset, n_samples, n_clusters)

        gauss(dataset, points, noise = 0.1, ax = axs[0])
        draw_line(dataset, points, (-1, 5), (0, -6.), 1.5, ax = axs[1])
        clustered_noise(dataset, points, 'circle', 0.1, ax = axs[2])
        

if __name__ == "__main__":
    np.random.seed(56)
    random.seed(56)
    run()