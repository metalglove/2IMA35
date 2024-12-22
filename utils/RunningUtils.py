from utils.GraphVisualizer import GraphVisualizer
from algorithms.BoruvkasAlgorithm import BoruvkasAlgorithm

def do_boruvka_on_graph(graph, graph_name, iterations = 10):
    gv = GraphVisualizer(graph)
    gv.plot_graph(graph_name)

    max_iterations = iterations
    plot_graph = lambda i: gv.plot_graph(f'{graph_name} boruvkas round {i}')
    alg = BoruvkasAlgorithm(graph, max_iterations, False, plot_graph)
    alg.run();


def get_runner_string(name, clean, dir):
    sub = 'clean'
    if not clean:
        sub = 'noisy'
    return f'{name}/{dir}/{sub}'
