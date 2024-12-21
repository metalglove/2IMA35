import math

from pyspark import SparkConf, SparkContext
from algorithms.Graph import edge_index


class BoruvkasAlgorithmSingleMachine:
    def __init__(self, max_iterations):
        self.max_iterations = max_iterations
        self.conf = SparkConf().setAppName('BoruvkaSingleMachine')
        self.sc = SparkContext.getOrCreate(conf=self.conf)

    def __contraction(self, V: dict[set], E, L):
        NGPrime = dict()
        VPrime = set()
        for u in V.keys():
            c = u
            v = u
            s = list()
            while v not in s:
                s = set(s).union([v])
                c = v
                v =  L[v]
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
            E = dict()
            V = dict({k: {}})
            print(f"final neighborhood: {k, NGPrime[k]}")
            return V, E

        print("neighborhoods: " + str([f"{ng} " for ng in NGPrime.items()]))

        def find_edges_out_of_neighborhood(leader, ng, E):
            print(f"find_edges_out_of_neighborhood: {str(leader)}: {str(ng)}")
            edges_going_out_neighborhood = dict()
            for (i, j) in E.keys():
                if (i not in ng and j not in ng) or (i in ng and j in ng):
                    continue

                wa = E[(i, j)]
                wb = math.inf
                if (i in ng and j not in ng):
                    edge = edge_index(leader, j)
                else: #(i not in ng or j in ng):
                    edge = edge_index(leader, i)

                # update the edge such that the vertex in the current neighborhood is updated to the leader
                if edge in edges_going_out_neighborhood:
                    wb = edges_going_out_neighborhood[edge]
                edges_going_out_neighborhood[edge] = min(wa, wb)

            print("\tout: " + str([f"{e} " for e in edges_going_out_neighborhood.items()]))

            # we emit the edges the edges that bridge the current neighborhood for the leader
            return edges_going_out_neighborhood

        def reduce_neighborhood_edges(x, y):
            edges_out = x | y
            return edges_out

        # map each neighborhood to contract their edges
        E = self.sc.parallelize(NGPrime) \
            .map(lambda leader: find_edges_out_of_neighborhood(leader, NGPrime[leader], E)) \
            .reduce(reduce_neighborhood_edges)

        def add_edge(i, j, w):
            edge = edge_index(i, j)
            E[edge] = w
            V[i] = V[i].union({j})
            V[j] = V[j].union({i})

        def remove_edge(i, j):
            edge = edge_index(i, j)
            del E[edge]
            if i in V:
                V[i] = V[i].difference({j})
            if j in V:
                V[j] = V[j].difference({i})

        def update_edge(a, b):
            # add an edge if it doesn't already exist
            # if it does already exist, check whether the weight needs to be updated (from a to b)

            if a == b:
                return
            if a not in E:
                return

            (i, j) = a
            wa = E[a]
            if b in E:
                E[b] = min(wa, E[b])
                remove_edge(i, j)
            else:
                remove_edge(i, j)
                (i, j) = b
                add_edge(i, j, wa)

        def find_key(i):
            for key, val in NGPrime.items():
                if i in val:
                    return key

            return -1

        # update edges to be their leaders
        for (i, j) in list(E.keys()):
            kp = find_key(i)
            lp = find_key(j)
            update_edge((i, j), edge_index(kp, lp))

        # generate new vertices dictionary
        V = dict()
        for (i, j) in E.keys():
            if i in V:
                V[i] = V[i].union({j})
            else:
                V[i] = {j}
            if j in V:
                V[j] = V[j].union({i})
            else:
                V[j] = {i}

        return V, E

    def __find_best_neighbors(self, V, E):
        def find_best_neighbor(x, vertices_it_connects_to):
            min_weight = math.inf
            best_vertex = x
            for j in vertices_it_connects_to:
                if x == j:
                    continue
                edge_weight = E[edge_index(x, j)]
                if edge_weight <= min_weight:
                    min_weight = edge_weight
                    best_vertex = j

            return x, best_vertex
        return dict(self.sc.parallelize(V.keys()) \
                .map(lambda k: find_best_neighbor(k, V[k])) \
                .collect())

        # Input: the graph G(V, E)
        # for each vertex u in G do
        #   if NG(u) = phi then
        #       Let Lambda(u) = u
        #   else
        #       Let Lambda(u) be the vertex v in N(u) whose weight is minimum;
        # end for
        # Output: For each vertex u in V, we return the mapping Lambda(u)

    def run(self, V, E):
        L = dict()
        n = len(V)
        Vs = dict()
        Es = dict()
        Vs[0] = V
        Es[0] = V
        if self.max_iterations > n:
            print(f"max_iterations > len(V): {self.max_iterations} > {n}")
            return

        for i in range(1, self.max_iterations + 1):
            print(f"round {i}")

            # find the nearest neighbor of each vertex
            L[i] = self.__find_best_neighbors(V, E)
            # contract the graph
            V, E = self.__contraction(V, E, L[i])
            Vs[i] = V
            Es[i] = E
            if len(V) == 1:
                break

        return L, Vs, Es
        # let i = 0 let leader(v) for each vertex v in V
        # repeat
        #   invoke FindNearestNeighbors(G(V, E)) that returns the mapping lambda: V -> V
        #   invoke Contractions(G(V, E)) that returns a graph G'(V', E'))
        #   let G(V, E) = G'(V', E')
        #   lambda_i = lambda
        #   i = i + 1
        # until |V| = 1
        # return a minimum spanning tree T of G(V, E)
