import math
from pyspark import SparkConf, SparkContext

# assumptions:
# - dataset has edges from all vertices to all vertices

# boruvka's algorithm

def edge_index(i, j):
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
            edge = edge_index(i, j)
            self.E[edge] = w
            self.V[i].append(j)
            self.V[j].append(i)

    def remove_edge(self, i, j):
        if i in self.V.keys() and j in self.V.keys():
            edge = edge_index(i, j)
            del self.E[edge]
            self.V[i].remove(j)
            self.V[j].remove(i)

    def update_edge(self, a, b):
        # add an edge if it doesn't already exist
        # if it does already exist, check whether the weight needs to be updated (from a to b)

        if a == b:
            return
        if a not in self.E:
            return

        (i, j) = a
        wa = self.E[a]
        if b in self.E:
            self.E[b] = min(wa, self.E[b])
            self.remove_edge(i, j)
        else:
            self.remove_edge(i, j)
            (i, j) = b
            self.add_edge(i, j, wa)


    def clean_vertices(self):
        self.V = {k: v for k, v in self.V.items() if len(v) > 0}

    def construct_components(self, VPrime, NGPrime):
        # VPrime -> Leader Vertices
        # NGPrime -> All nodes in a component (all vertices that should be collapsed to one vertex)

        # NGPrime[2] = {1, 2, 3, 6} -> 2 is the leader, so all remaining vertices should collapse to 2.
        # all edges that are inside that component should be deleted. ALL edges that are bridging THIS
        # component to another, have to be tracked such that we can find the minimum cost to bridge the
        # components.

        # delete non-leader vertices and edges of each component
        # for multiple edges that have the same leaders, choose one with minimum weight
        # all edges leaving each component will be incident to the leader of the component
        # return the contracte graph

        # alternative idea:
        # create a completely new graph G2 with (V2 and E2)
        # for every key in ngprime: create a new vertex where i = key and add it to G2
        # create a n*n table where n = |V2|. Set all values in here to inf
        # loop over all edges in G:
        #   determine using NGPrime if the edge is trans-cluster.
        #   if it's a trans-cluster edge, compare its weight to table[i][j]. make sure that there's no redundancy with table[j][i] (check for largest)
        #   if this weight is smaller than table[i][j], set table[i][j] to the weight of the current edge. else, nothing
        # loop over all i and j indicies:
        #   check if the value of table[i][j] is not equal to inf
        #   if so, create an edge from vertex i to vertex j in G2. else, nothing


        # collapse vertices to their leader
        for vp in VPrime:
            neighborhood = NGPrime[vp]

            # find edges of each vertex in the neighborhood
            edges = set([edge_index(u,v) for v in neighborhood for u in self.V[v]])

            # filter edges that are going out of the component
            for (u, v) in edges:
                edge = (u, v)
                if u not in neighborhood or v not in neighborhood:
                    if u in neighborhood:
                        u = vp
                    else:
                        v = vp
                    # update an edge bridging the components to their leader vertex
                    self.update_edge(edge, edge_index(u, v))
                else:
                    # remove edge
                    self.remove_edge(u, v)

        # remove vertices that are in the neighborhood
        self.clean_vertices()

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
            self.G.V = dict({k: []})
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

    def run(self):
        print(f"running boruvkas algorithm: max_iterations = {self.max_iterations}\nGraph:")
        self.__print_graph()
        self.__reset()
        L = dict()
        for i in range(1, self.max_iterations + 1):
            print(f"round {i}")
            L[i] = self.__find_nearest_neighbors()
            self.__contraction(L[i])
            self.__print_graph()
            if len(self.G.V) <= 1:
                break
        return self.G

    def __print_graph(self):
        print(f"\tvertices:" + str([f"{v} " for v in self.G.V.items()]) +"\n\tedges:" + str([f"{e} " for e in self.G.E.items()]))

from operator import concat

class BoruvkasAlgorithmSingleMachine:
    def __init__(self, max_iterations):
        self.max_iterations = max_iterations
        self.conf = SparkConf().setAppName('BoruvkaSingleMachine_MST')
        self.sc = SparkContext.getOrCreate(conf=self.conf)

    def __contraction(self, V: dict[list], E, L):
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
            V = dict({k: []})
            print(f"final neighborhood: {k, NGPrime[k]}")
            return V, E

        print("neighborhoods: " + str([f"{ng} " for ng in NGPrime.items()]))

        def find_edges_in_and_out_of_neighborhood(leader, ng, E):
            print("find_edges_in_and_out_of_neighborhood: " + str(leader) + str(ng))

            edges_in_neighborhood = dict()
            edges_going_out_neighborhood = dict()
            for (i, j) in E.keys():
                if i not in ng and j not in ng:
                    continue
                if i in ng and j in ng:
                    edges_in_neighborhood[(i, j)] = E[(i, j)]
                    continue

                wa = E[(i, j)]
                wb = math.inf
                if (i in ng and j not in ng):
                    edge = edge_index(leader, j)
                else: #(i not in ng or j in ng):
                    edge = edge_index(i, leader)
                
                if edge in edges_going_out_neighborhood:
                    wb = edges_going_out_neighborhood[edge]
                edges_going_out_neighborhood[edge] = min(wa, wb)

            print("\tin: " + str([f"{e} " for e in edges_in_neighborhood.items()]))
            print("\tout: " + str([f"{e} " for e in edges_going_out_neighborhood.items()]))

            # we emit the edges the edges that are in and out the neighborhood for the leader
            return set([leader]), edges_in_neighborhood, edges_going_out_neighborhood, {leader: ng}

        def find_edges_out_of_neighborhood(leader, ng, E):
            print("find_edges_out_of_neighborhood: " + str(leader) + str(ng))

            edges_going_out_neighborhood = dict()
            connected_vertices = set()

            for (i, j) in E.keys():
                if (i not in ng and j not in ng) or (i in ng and j in ng):
                    continue

                wa = E[(i, j)]
                wb = math.inf
                if (i in ng and j not in ng):
                    edge = edge_index(leader, j)
                    connected_vertices = set(connected_vertices).union({j})
                else: #(i not in ng or j in ng):
                    edge = edge_index(i, leader)
                    connected_vertices = set(connected_vertices).union({i})
                
                if edge in edges_going_out_neighborhood:
                    wb = edges_going_out_neighborhood[edge]
                edges_going_out_neighborhood[edge] = min(wa, wb)

            print("\tout: " + str([f"{e} " for e in edges_going_out_neighborhood.items()]))

            # we emit the edges the edges that are in and out the neighborhood for the leader
            return set([leader]), edges_going_out_neighborhood, {leader: connected_vertices}

        def reduce_contraction(x, y):
            leaders = x[0] | y[0]
            edges_out = x[1] | y[1]
            connected_vertices = x[2] | y[2]
            edges_out2 = edges_out.copy()

            def find_key(i):
                for key, val in connected_vertices.items():
                    if i in val:
                        return key

                return -1
                # k = [key for key, val in connected_vertices.items() if i in val][0]

            for l in leaders:
                edges_out = edges_out2.copy()
                for (i, j), w in edges_out.items():
                    # if both of the vertices are leaders, continue
                    if (i == l and j in leaders) or (j == l and i in leaders):
                        continue
                    wb = math.inf
                    if (i == l):
                        k = find_key(j)
                        edge = edge_index(i, k)
                    else: #(i not in ng or j in ng):
                        k = find_key(i)
                        edge = edge_index(j, k)
                    
                    # the leader for k has not yet been added to the connected vertices dict.
                    if k == -1:
                        continue

                    if edge in edges_out2:
                        wb = edges_out2[edge]

                    del edges_out2[(i, j)]
                    edges_out2[edge] = min(w, wb)

            return leaders, edges_out2, connected_vertices
        
        # map each neighborhood to contract their edges
        (V, E, _) = self.sc.parallelize(NGPrime) \
            .map(lambda leader: find_edges_out_of_neighborhood(leader, NGPrime[leader], E)) \
            .reduce(reduce_contraction)
            # .map(lambda leader: find_edges_in_and_out_of_neighborhood(leader, NGPrime[leader], E)) \

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

        for i in range(1, self.max_iterations + 1):
            # find the nearest neighbor of each vertex
            L[i] = self.__find_best_neighbors(V, E)
            # contract the graph
            V, E = self.__contraction(V, E, L[i])
            


        #rdd = sc.parallelize(adj)


        # let i = 0 let leader(v) for each vertex v in V
        # repeat
        #   invoke FindNearestNeighbors(G(V, E)) that returns the mapping lambda: V -> V
        #   invoke Contractions(G(V, E)) that returns a graph G'(V', E'))
        #   let G(V, E) = G'(V', E')
        #   lambda_i = lambda
        #   i = i + 1
        # until |V| = 1
        # return a minimum spanning tree T of G(V, E)
        pass

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
    max_iterations = 10
    # alg = BoruvkasAlgorithm(G, max_iterations)
    alg = BoruvkasAlgorithmSingleMachine(max_iterations)
    alg.run(G.V, G.E)

if __name__ == "__main__":
    main()