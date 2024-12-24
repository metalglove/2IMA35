# AffinityClusteringMPC.py

import math
from pyspark.context import SparkContext
from pyspark.conf import SparkConf

from algorithms.Graph import edge_index

# AffinityClustering is the MPC implementation of Boruvka's algorithm.
# That is possible because Boruvka's algorithm is parallel friendly in nature.
class AffinityClusteringMPC():
    def __init__(self):
        self.conf = SparkConf().setAppName('AffinityClustering')
        self.sc = SparkContext.getOrCreate(conf=self.conf)

    # def __find_nearest_neighbors(self):
    #     # Find the nearest neighbors of each vertex.
    #     L = self.sc.broadcast({ i : self.__find_nearest_neighbor(i) for i in self.V })
    #     return L

    def run(self, V, E):
        # to avoid referncing an object that has the SparkContext, we add the methods
        # within the run method.
        def find_nearest_neighbor_map(u: int, V: dict[int, set], E: dict[tuple[int, int], int]) -> tuple[int, int]:
            '''
            Input: A graph G(V, E).
            Output: The mapping L: V -> V of the input graph.
            '''
            # if the vertex has no edges, then the vertex itself is the only suitable mapping.
            if not any(V[u]):
                return u, u
            # otherwise, find the nearest neighbor of u by minimizing weight.
            else:
                (w, v) = min((E[edge_index(u, v)], v) for v in V if edge_index(u, v) in E)
            return u, v

        def find_nearest_neighbor_reduce(a: dict[int, int], b: dict[int, int]) -> dict[int, int]:
            return a | b

        Lambda = self.sc.parallelize(V) \
            .map(lambda v: find_nearest_neighbor_map(v, V, E)) \
            .reduceByKey(find_nearest_neighbor_reduce) \
            .collectAsMap()
            
        print(f'Lambda: {Lambda}')
        
        def create_neighborhood_map(u: int, Lambda: dict[int, int]) -> tuple[int, set]:
            '''
            Input: The graph G(V,E) and the mapping lambda: V' -> V
            Output: A contracted graph G'(V', E') of the input graph G(V,E).
            '''
            c = u
            v = u
            S = set()
            while v not in S:
                S.add(v)
                c = v
                v = Lambda[v]
            return min(c, v), S

        def create_neighborhood_reduce(a: set[int], b: set[int]) -> set[int]:
            return a | b

        NG = self.sc.parallelize(V) \
                 .map(lambda v: create_neighborhood_map(v, Lambda)) \
                 .reduceByKey(create_neighborhood_reduce) \
                 .collectAsMap()

        print(f'Neighborhoods: {NG}')
        
        def contract_neighborhood_map(leader: int, neighborhood: set, E: dict[tuple[int, int], int]) -> tuple[int, dict[tuple[int, int], int]]:
            '''
            
            '''
            edges_going_out_neighborhood = dict()
            updates = dict()
            for (i, j) in E.keys():
                if (i not in neighborhood and j not in neighborhood) or (i in neighborhood and j in neighborhood):
                    continue
                
                wa = E[(i, j)]
                wb = math.inf
                if (i in neighborhood and j not in neighborhood):
                    edge = edge_index(leader, j)
                else: #(i not in ng or j in ng):
                    edge = edge_index(leader, i)
                
                # update the edge such that the vertex in the current neighborhood is updated to the leader
                if edge in edges_going_out_neighborhood:
                    wb = edges_going_out_neighborhood[edge]
                
                w = min(wa, wb)
                edges_going_out_neighborhood[edge] = w

                # track update
                updates[(i, j)] = (leader, w) 

            # we emit the edges that bridge the current neighborhood for the leader
            return updates

        def contract_neighborhood_reduce(a, b):
            keys = set(a.keys()) | set(b.keys())
            result_dict = dict()
            for edge in keys:
                if edge in a and edge in b:
                    (leader_a, weight) = a[edge]
                    (leader_b, weight) = b[edge]
                    result_dict[edge_index(leader_a, leader_b)] = weight
                elif edge in a:
                    result_dict[edge] = a[edge]
                else:
                    result_dict[edge] = b[edge]
            return result_dict

        contracted_graph = self.sc.parallelize(NG) \
            .map(lambda leader: contract_neighborhood_map(leader, NG[leader], E)) \
            .reduce(contract_neighborhood_reduce)
        
        print(f'contracted_graph: {contracted_graph}')
        

        # self.sc.parallelize(E) \
        #     .map(lambda e: ((e[0], e[1]), 1)) \
        #     .reduceByKey(lambda a, b: a + b).collect()
        
# let i = 0 let leader(v) for each vertex v in V
# repeat
#   invoke FindNearestNeighbors(G(V, E)) that returns the mapping lambda: V -> V
#   invoke Contractions(G(V, E)) that returns a graph G'(V', E'))
#   let G(V, E) = G'(V', E')
#   lambda_i = lambda
#   i = i + 1
# until |V| = 1
# return a minimum spanning tree T of G(V, E)