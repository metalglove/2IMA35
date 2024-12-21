from algorithms.Graph import Graph


class Algorithm:
    def __init__(self, G: Graph, max_iterations: int):
        self.__G = G
        self.max_iterations = max_iterations
        self.reset()

    def reset(self):
        self.G = self.__G

    def print_graph(self):
        print(f"\tvertices:" + str([f"{v} " for v in self.G.V.items()]) +"\n\tedges:" + str([f"{e} " for e in self.G.E.items()]))

