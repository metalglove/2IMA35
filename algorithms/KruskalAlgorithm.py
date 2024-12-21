import math
from algorithms.Algorithm import Algorithm
from algorithms.Graph import Graph, edge_index

# SingleLinkAgglomerativeClusteringAlgorithm
class KruskalAlgorithm(Algorithm):
    def __init__(self, G: Graph, max_iterations: int):
        super().__init__(G, max_iterations)

    
    def merge(self):
        keys = list(self.G.V.keys())
        # for each vertex merge to the closest one
        for i in range(len(keys)):
            key = keys[i]
            # vertex merged previously (possibly, i - 1)
            if key not in self.G.V:
                continue
            
            # find vertex with shortest distance from i
            v = self.G.V[key]
            min_edge_w = math.inf
            vertex_shortest_distance = None
            for u in v:
                edge = edge_index(key, u)
                w = self.G.E[edge]
                if w < min_edge_w:
                    vertex_shortest_distance = u
                    min_edge_w = w

            # remove that vertex
            self.G.remove_vertex(vertex_shortest_distance)

    def run(self, print_graph = False, plot_graph = None):
        print(f"running Kruskal algorithm: max_iterations = {self.max_iterations}")
        self.reset()

        centroids = None
        for i in range(1, self.max_iterations + 1):
            print(f"round {i}")

            self.merge()

            if print_graph:
                self.print_graph()
            if plot_graph is not None:
                plot_graph(i)
            if len(self.G.V) == 1:
                break