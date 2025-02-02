import numpy as np
from sklearn.cluster import KMeans

from algorithms.Algorithm import Algorithm
from algorithms.Graph import Graph
from utils.GraphUtils import frobenius


class KMeansAlgorithm(Algorithm):
    def __init__(self, G: Graph, max_iterations: int, n_clusters: int, print_graph: bool = False, plot_graph = None):
        super().__init__(G, max_iterations, print_graph, plot_graph)
        self.reset()
        self.n_clusters = n_clusters
        self.previous = None

    def run(self):
        print(f"running kmeans algorithm: max_iterations = {self.max_iterations}")
        self.reset()

        centroids = None
        for i in range(1, self.max_iterations + 1):
            print(f"round {i}")

            kmeans = KMeans(n_clusters=self.n_clusters, 
                            max_iter=1, 
                            n_init=1,
                            init=(centroids if centroids is not None else 'k-means++'),
                            random_state=0)

            kmeans = kmeans.fit(np.array(self.G.points))
            centroids = kmeans.cluster_centers_

            if self.do_print_graph:
                self.print_graph(centroids)
            if self.plot_graph is not None:
                self.plot_graph(i, centroids)
            if not self.have_centroids_moved(centroids):
                break

        return centroids
    
    def print_graph(self, centroids):
        print(f"centroids: {centroids}")

    def have_centroids_moved(self, centroids):
        if self.previous is None:
            self.previous = sum([frobenius(centroid) for centroid in centroids])
            return True
        
        current = sum([frobenius(centroid) for centroid in centroids])
        diff = abs(current - self.previous)
        print(f'frobenius norm | previous: {self.previous}, current: {current}, diff: {diff}')
        self.previous = current
        return diff >= 0.0001
        