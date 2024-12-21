
from algorithms import BoruvkasAlgorithm
from algorithms.BoruvkasAlgorithmSingleMachine import BoruvkasAlgorithmSingleMachine
from algorithms.Graph import Graph


def main():
    G = Graph()
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
    # G.add_edge(1, 7, 8)
    # G.add_edge(1, 2, 2)
    # G.add_edge(2, 3, 1)
    # G.add_edge(2, 6, 8)
    # G.add_edge(3, 7, 7)
    # G.add_edge(3, 4, 6)
    # G.add_edge(3, 6, 7)
    # G.add_edge(4, 5, 1)
    # G.add_edge(4, 7, 8)
    # G.add_edge(4, 10, 5)
    # G.add_edge(5, 6, 8)
    # G.add_edge(7, 9, 4)
    # G.add_edge(8, 7, 3)
    # G.add_edge(8, 9, 3)
    # G.add_edge(8, 10, 2)
    # G.add_edge(10, 9, 4)

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
    G.add_edge(1, 2, 3)
    G.add_edge(2, 3, 2)
    G.add_edge(3, 4, 1)
    G.add_edge(4, 5, 2)
    G.add_edge(5, 7, 3)
    G.add_edge(7, 8, 5)
    G.add_edge(8, 6, 3)
    G.add_edge(6, 9, 2)
    G.add_edge(9, 10, 1)
    G.add_edge(10, 1, 2)
    max_iterations = 10
    # alg = BoruvkasAlgorithm(G, max_iterations)
    alg = BoruvkasAlgorithmSingleMachine(max_iterations)
    alg.run(G.V, G.E)

if __name__ == "__main__":
    main()