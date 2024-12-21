from algorithms.Graph import Graph


class Algorithm:
    def __init__(self, G: Graph, max_iterations: int, print_graph: bool, plot_graph):
        self.__G = G
        self.max_iterations = max_iterations
        self.do_print_graph = print_graph
        self.plot_graph = plot_graph
        self.reset()

    def reset(self):
        self.G = self.__G

    def run(self):
        raise NotImplementedError('Use the implementation of an algorithm')

    def print_graph(self):
        print(f"\tvertices:" + str([f"{v} " for v in self.G.V.items()]) +"\n\tedges:" + str([f"{e} " for e in self.G.E.items()]))

