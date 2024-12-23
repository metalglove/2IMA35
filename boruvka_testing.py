from algorithms.Graph import Graph
from utils.datasets import generate_points
from utils.GraphGenerator import GraphGenerator
from utils.GraphVisualizer import GraphVisualizer
from algorithms.BoruvkasAlgorithm import BoruvkasAlgorithm

def run():
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

    max_iterations = 10
    alg = BoruvkasAlgorithm(G, max_iterations, True)
    alg.run()

if __name__ == '__main__':
    run()