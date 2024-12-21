
def edge_index(i, j):
    if i > j:
        return (i, j)
    return (j, i)


class Graph:
    def __init__(self):
        self.V = dict()
        self.E = dict()
        self.C = dict()

    def add_vertex(self, i):
        self.V[i] = set()

    def remove_vertex(self, i):
        if i not in self.V.keys():
            return
        if len(self.V[i]) > 0:
            # remove vertex from all edges to that vertex
            keys = list(self.V[i])
            for j in range(len(keys)):
                self.remove_edge(i, keys[j])
        del self.V[i]
        
    def add_edge(self, i, j, w):
        if i in self.V.keys() and j in self.V.keys():
            edge = edge_index(i, j)
            self.E[edge] = w
            self.V[i] = self.V[i] | {j}
            self.V[j] = self.V[j] | {i}

    def remove_edge(self, i, j):
        if i in self.V.keys() and j in self.V.keys():
            edge = edge_index(i, j)
            del self.E[edge]
            self.V[i] = self.V[i] - {j}
            self.V[j] = self.V[j] - {i}

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
