from algorithms.Algorithm import Algorithm
from algorithms.Graph import Component, Graph, edge_index

class SingleLinkAgglomerativeClusteringAlgorithm(Algorithm):
    def __init__(self, G: Graph, max_iterations: int):
        super().__init__(G, max_iterations)


    def __find_component(self, v) -> Component:
        return self.component_map[v]
    
    def __merge(self):
        ic = 0
        jc = 0

        # merge components (vertices) that share the minimum edge (across components)
        while ic == jc:
            if len(self.edges) == 0:
                return
            edge = self.edges[0]
            self.edges.remove(edge)
            ((i, j), w) = edge
            ic = self.__find_component(i)
            jc = self.__find_component(j)

        self.G.components[ic] = self.G.components[ic].expand(self.G.components[jc])
        self.G.components[jc] = None
        for i in self.G.components[ic].V:
            self.component_map[i] = ic

    def run(self, print_graph = False, plot_graph = None):
        print(f"running Single-Linkage Agglomerative algorithm: max_iterations = {self.max_iterations}")
        self.reset()

        # sort edges in ascending order based on their weight
        self.edges = sorted(list(self.G.E.items()), key = lambda x: x[1])
        self.G.components = list[Component]()
        
        for v in self.G.V.keys():
            self.G.components.append(Component(v))

        self.component_map = dict()
        for i in range(len(self.G.components)):
            self.component_map[i] = i

        # while the number of found components is not 1 or max_iterations is not reached, repeat
        for i in range(1, self.max_iterations + 1):

            self.__merge()

            num_components = len([component for component in self.G.components if type(component) is Component])

            if print_graph:
                self.print_graph()

            if num_components <= 5:
                if plot_graph is not None:
                    plot_graph(i)

            if num_components == 1:
                print(f"round {i}: components 1")
                break