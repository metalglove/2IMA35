import math
# assumptions:
# - dataset has edges from all vertices to all vertices

# boruvka's algorithm

def node_index(i, j):
    if i > j:
        return (i, j)
    return (j, i)

class Component:
    def __init__(self, leader, V, E):
        self.leader = leader
        self.V = V
        self.E = E

class Graph:
    def __init__(self):
        self.V = dict()
        self.E = dict()
        self.C = dict()

    def add_vertex(self, i):
        self.V[i] = list()

    def add_edge(self, i, j, w):
        if i in self.V.keys() and j in self.V.keys():
            edge = node_index(i, j)
            self.E[edge] = w
            self.V[i].append(j)
            self.V[j].append(i)

    def remove_edge(self, i, j):
        if i in self.V.keys() and j in self.V.keys():
            edge = node_index(i, j)
            if edge in self.E:
                del self.E[edge]
                self.V[i].remove(j)
                self.V[j].remove(i)

    def construct_components(self, VPrime, NGPrime, NGPrimeEdges):
        flat_map = lambda xs: [y for xy in xs for y in xy]
        prime_edges = set(flat_map(NGPrimeEdges.values()))
        leader_edges = set([(u, v) for (u, v) in prime_edges if u in VPrime and v in VPrime])
        reduced_edges = set(self.E.keys()) - prime_edges
        edges_outside_components = set(reduced_edges) | leader_edges

        # removing unnessacery edges.
        edges_to_remove = set(self.E.keys()) - edges_outside_components
        for (i, j) in edges_to_remove:
            self.remove_edge(i, j)

        # NOTE: dumbest thing ever to remove edges that are in the reduced edges set
        # which are not in the prime edges set but in a component
        for (u, v) in reduced_edges:
            if u not in VPrime and v not in VPrime:
                for prime in NGPrime.values():
                    b = set([u, v])
                    a = set(prime) - b 
                    if len(prime) - 2 == len(a):
                        edges_outside_components.remove((u, v))
                        continue
            elif (u, v) not in prime_edges:
                edges_outside_components.remove((u, v))

        # WIP: need to figure out how to filter the edges such that only the edges
        # outside of the components remain. Then, we can check per component which
        # edges we need to keep.
        # It is also important to create/update edges for the ones between components
        # as the leaders will then be connected with the minimum edge weight connecting
        # the components.
        # Next, we need to remove all vertices that are no longer used.


        # all vertices have the same leader will be one component
        for (leader, neighborhood) in NGPrime.items():
            for i in neighborhood:
                for (u, v) in edges_outside_components:
                    if u == i or v == i:
                        pass
                
        # check G.V for empty vertex list -> remove the vertex

            
            

        # delete non-leader vertices and edges of each component
        # for multiple edges that have the same leaders, choose one with minimum weight
        # all edges leaving each component will be incident to the leader of the component
        # return the contracte graph
        pass


class BoruvkasAlgorithmSingleMachine:
    def __init__(self, G: Graph):
        self.__G = G
        self.__reset()

    def __reset(self):
        self.i = 0
        self.v = 0
        self.G = self.__G

    def __contraction(self, L):
        NGPrime = dict()
        NGPrimeEdges = dict()
        VPrime = list()
        for u in self.G.V:
            c = u
            v = u
            s = list()
            E = list()
            while v not in s:
                s = set(s).union([v])
                c = v
                v_ = L[v]
                e = node_index(v_, v)
                v = v_
                E.append(e)
            c = min(c, v)
            if c in VPrime:
                NGPrime[c] = set(NGPrime[c]).union(s)
                NGPrimeEdges[c] = set(NGPrimeEdges[c]).union(E)
            else:
                VPrime = set(VPrime).union([c])
                NGPrime[c] = set([c]).union(s)
                NGPrimeEdges[c] = set(E)
        print("hi")
        self.G.construct_components(VPrime, NGPrime, NGPrimeEdges)
        return self.G

    def __find_nearest_neighbors(self):
        L = dict()
        for u in self.G.V:
            min_weight = math.inf
            best_vertex = u
            for j in self.G.V[u]:
                if u == j:
                    continue
                edge_weight = self.G.E[node_index(u, j)]
                if edge_weight <= min_weight:
                    min_weight = edge_weight
                    best_vertex = j
            L[u] = best_vertex
        return L

    def run(self):
        self.__reset()
        i = 0
        L = dict()
        for v in self.G.V:
            L[i] = self.__find_nearest_neighbors()
            self.G = self.__contraction(L[i])
            i = i + 1
            if len(self.V) <= 1:
                break
        return self.G

def main():
    G = Graph()
    G.add_vertex(1)
    G.add_vertex(2)
    G.add_vertex(3)
    G.add_vertex(4)
    G.add_vertex(5)
    G.add_vertex(6)
    G.add_vertex(7)
    G.add_vertex(8)
    G.add_vertex(9)
    G.add_vertex(10)
    G.add_edge(1, 7, 8)
    G.add_edge(1, 2, 2)
    G.add_edge(2, 3, 1)
    G.add_edge(2, 6, 8)
    G.add_edge(3, 7, 7)
    G.add_edge(3, 4, 6)
    G.add_edge(3, 6, 7)
    G.add_edge(4, 5, 1)
    G.add_edge(4, 7, 8)
    G.add_edge(4, 10, 5)
    G.add_edge(5, 6, 8)
    G.add_edge(7, 9, 4)
    G.add_edge(8, 7, 3)
    G.add_edge(8, 9, 3)
    G.add_edge(8, 10, 2)
    G.add_edge(10, 9, 4)

    # G.add_vertex(1)
    # G.add_vertex(2)
    # G.add_vertex(3)
    # G.add_vertex(4)
    # G.add_vertex(5)
    # G.add_vertex(6)
    # G.add_vertex(7)
    # G.add_vertex(8)
    # G.add_vertex(9)
    # G.add_vertex(10)
    # G.add_edge(1, 2, 3)
    # G.add_edge(2, 3, 2)
    # G.add_edge(3, 4, 1)
    # G.add_edge(4, 5, 2)
    # G.add_edge(5, 7, 3)
    # G.add_edge(7, 8, 5)
    # G.add_edge(8, 6, 3)
    # G.add_edge(6, 9, 2)
    # G.add_edge(9, 10, 1)
    # G.add_edge(10, 1, 2)

    alg = BoruvkasAlgorithmSingleMachine(G)
    alg.run()

if __name__ == "__main__":
    main()