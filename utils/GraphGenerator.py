import math
import sys

import numpy as np

from algorithms.Graph import Graph
from utils.GraphUtils import euclidean_distance, find_min_max_x_y

class GraphGenerator(object):

    def __init__(self, points: list[tuple[2]], edges: list[tuple[3]] = []):
        self.points = points
        self.edges = edges

    def add_gaussian_noise(self, noise=0.1):
        min_x, max_x, min_y, max_y = find_min_max_x_y(self.points)

        size = int(np.ceil(len(self.points) * noise))
        for i in range(size):
            x = np.random.uniform(min_x, max_x)
            y = np.random.uniform(min_y, max_y)
            self.points.append((x, y))

        return self

    def add_clustered_noise(self, type='', noise=0.1):
        min_x, max_x, min_y, max_y = find_min_max_x_y(self.points)

        size = int(np.ceil(len(self.points) * noise))
        if type == '':
            raise ValueError('No parameter given for type of clustered noise, \n\ttry: circle, horizontal_line or vertical_line')
        elif type == 'circle':
            center = ((max_x + min_x) / 2, (max_y + min_y) / 2)
            radius = min(max_x - min_x, max_y - min_y) * 0.7 / 2
            pi = math.pi
            for i in range(size):
                x = math.cos(2 * pi / size * i) * radius + center[0]
                y = math.sin(2 * pi / size * i) * radius + center[1]
                self.points.append((x, y))
        elif type == 'horizontal_line':
            y_center = (max_y + min_y) / 2
            interval = (max_x - min_x) / size
            current_x = min_x
            for i in range(size):
                self.points.append((current_x, y_center))
                current_x += interval
        elif type == 'vertical_line':
            x_center = (max_x + min_x) / 2
            interval = (max_y - min_y) / size
            current_y = min_y
            for i in range(size):
                self.points.append((x_center, current_y))
                current_y += interval
        else:
            NotImplementedError('%s not implemented' % type)

        return self

    def add_point(self, point) -> Graph:
        self.points.append(point)

        return self

    def add_points(self, points) -> Graph:
        for point in points:
            self.add_point(point)

        return self

    def draw_line(self, start_point, end_point, spacing) -> Graph:
        end_right = start_point[0] < end_point[0]
        end_above = start_point[1] < end_point[1]

        current_point = list(start_point)
        x_vals = []
        y_vals = []

        x_dist = abs(start_point[0] - end_point[0])
        y_dist = abs(start_point[1] - end_point[1])

        x_spacing = (x_dist / (x_dist + y_dist)) * spacing
        y_spacing = (y_dist / (x_dist + y_dist)) * spacing

        if (end_right):
            while (current_point[0] <= end_point[0]):
                current_point[0] += x_spacing
                x_vals.append(current_point[0])
        else:
            while (current_point[0] >= end_point[0]):
                current_point[0] -= x_spacing
                x_vals.append(current_point[0])

        if (end_above):
            while (current_point[1] <= end_point[1]):
                current_point[1] += y_spacing
                y_vals.append(current_point[1])
        else:
            while (current_point[1] >= end_point[1]):
                current_point[1] -= y_spacing
                y_vals.append(current_point[1])

        for i in range(max(len(x_vals), len(y_vals))):
            if (i >= len(x_vals)):
                self.add_point((current_point[0], y_vals[i]))
            elif (i >= len(y_vals)):
                self.add_point((x_vals[i], current_point[1]))
            else:
                self.add_point((x_vals[i], y_vals[i]))

        return self

    def to_graph(self, gen_pair_wise = False, f = None) -> Graph:
        G = Graph()
        G.points = self.points

        # if no distance function is provided, we assume euclidean distance
        G.distance_function = euclidean_distance

        # add a vertex for each point (map vertex id to xy data from points list)
        for i in range(len(G.points)):
            G.add_vertex(i)

        if gen_pair_wise:
            # add an edge for each possible point
            for i in G.V:
                for j in G.V:
                    if i == j:
                        continue

                    w = G.distance_function(G.points[i], G.points[j])
                    G.add_edge(i, j, w)

        # add any remaining edges the user already specified (override any generated edges if needed)
        if len(self.edges) > 0:
            for edge in self.edges:
                if len(edge) == 3:
                    (i, j, w) = edge
                elif len(edge) == 2:
                    (i, j) = edge
                    w = G.distance_function(G.points[i], G.points[j])
                else:
                    print(f"edge not a tuple of length 2 or 3: {edge}")
                    continue

                G.add_edge(i, j, w)

        return G