from algorithms.Algorithm import Algorithm
from algorithms.Graph import Graph, edge_index
import math

class BoruvkasAlgorithm(Algorithm):
    def __init__(self, G: Graph, max_iterations, print_graph = False, plot_graph = None):
        super().__init__(G, max_iterations, print_graph, plot_graph)

    def __contraction(self, L):
        NGPrime = dict()
        VPrime = set()

        for u in self.G.V:
            c = u
            v = u
            s = list()
            while v not in s:
                s = set(s).union([v])
                c = v
                v = L[v]
            c = min(c, v)
            # if c is a leader (c in V')
            if c in VPrime:
                # update the vertices in the neighborhood of c
                NGPrime[c] = set(NGPrime[c]).union(s)
            else:
                # otherwise, add to the neighborhood and V'
                VPrime = set(VPrime).union([c])
                NGPrime[c] = set([c]).union(s)

        # we have finished clustering to a single vertex.
        if len(NGPrime.keys()) == 1:
            k = next(iter(NGPrime))
            self.G.E = dict()
            self.G.V = dict({k: {}})
            print(f"final neighborhood: {k, NGPrime[k]}")
            return

        print("neighborhoods: " + str([f"{ng} " for ng in NGPrime.items()]))
        self.G.construct_components(VPrime, NGPrime)
        return NGPrime

    def __find_nearest_neighbors(self):
        L = dict()
        for u in self.G.V:
            min_weight = math.inf
            best_vertex = u
            for j in self.G.V[u]:
                if u == j:
                    continue
                edge_weight = self.G.E[edge_index(u, j)]
                if edge_weight <= min_weight:
                    min_weight = edge_weight
                    best_vertex = j
            L[u] = best_vertex
        return L

    def run(self):
        print(f"running boruvkas algorithm: max_iterations = {self.max_iterations}\nGraph:")
        if self.do_print_graph:
            self.print_graph()
        self.reset()
        L = dict()
        Vs = dict()
        Es = dict()
        NGs = dict()
        Vs[0] = self.G.V
        Es[0] = self.G.E
        NGs[0] = self.G.V
        for i in range(1, self.max_iterations + 1):
            print(f"round {i}")
            L[i] = self.__find_nearest_neighbors()
            NGs[i] = self.__contraction(L[i])
            Vs[i] = self.G.V
            Es[i] = self.G.E
            if self.do_print_graph:
                self.print_graph()
            if self.plot_graph is not None:
                self.plot_graph(i)
            if len(self.G.V) <= 1:
                break
        return L, Vs, Es, NGs

