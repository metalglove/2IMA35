import math
import sys
import matplotlib.pyplot as plt
from algorithms.Graph import Graph

def find_min_max_x_y(points):
    max_x, max_y = -1.0, -1.0
    min_x, min_y = sys.maxsize, sys.maxsize

    for v in points:
        max_x = max(max_x, v[0])
        max_y = max(max_y, v[1])
        min_x = min(min_x, v[0])
        min_y = min(min_y, v[1])

    return min_x, max_x, min_y, max_y

def frobenius(p1):
    res = p1[0]**2
    res += p1[1]**2
    return math.sqrt(res)

def euclidean_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def generate_graph(points, f):
    G = Graph()
    G.points = points

    # add a vertex for each point (map vertex id to xy data from points list)
    for i in range(len(points)):
        G.add_vertex(i)

    # add an edge for each possible point
    for i in G.V:
        for j in G.V:
            if i == j:
                continue

            w = f(G.points[i], G.points[j])
            G.add_edge(i, j, w)

    return G

def generate_plot_for_graph(G):

    plt.figure()

    xs, ys = [], []
    for x in G.V.keys():
        xs.append(G.points[x][0])
        ys.append(G.points[x][1])

    plt.scatter(xs, ys, c='r')

def plot_graph(V, E, points, ax):
    xs, ys = [], []
    for x in V.keys():
        xs.append(points[x][0])
        ys.append(points[x][1])

    # print(xs, ys)
    ax.scatter(xs, ys, c='r')