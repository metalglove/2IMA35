from algorithms.Graph import Graph, edge_index
import math

class BoruvkasAlgorithm:
    def __init__(self, G: Graph, max_iterations):
        self.__G = G
        self.__reset()
        self.max_iterations = max_iterations

    def __reset(self):
        self.i = 0
        self.v = 0
        self.G = self.__G

    def __contraction(self, L):
        NGPrime = dict()
        VPrime = set()

        for u in self.G.V:
            c = u
            v = u
            s = list()
            E = list()
            while v not in s:
                s = set(s).union([v])
                c = v
                v_ = L[v]
                e = edge_index(v_, v)
                v = v_
                E.append(e)
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

    def run(self, print_graph = False):
        print(f"running boruvkas algorithm: max_iterations = {self.max_iterations}\nGraph:")
        if print_graph:
            self.__print_graph()
        self.__reset()
        L = dict()
        Vs = dict()
        Es = dict()
        Vs[0] = self.G.V
        Es[0] = self.G.E
        for i in range(1, self.max_iterations + 1):
            print(f"round {i}")
            L[i] = self.__find_nearest_neighbors()
            self.__contraction(L[i])
            Vs[i] = self.G.V
            Es[i] = self.G.E
            if print_graph:
                self.__print_graph()
            if len(self.G.V) <= 1:
                break
        return L, Vs, Es

    def __print_graph(self):
        print(f"\tvertices:" + str([f"{v} " for v in self.G.V.items()]) +"\n\tedges:" + str([f"{e} " for e in self.G.E.items()]))
