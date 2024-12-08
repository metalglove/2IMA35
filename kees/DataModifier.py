import math
import sys

import numpy as np
from sklearn.datasets import make_circles


class DataModifier:

    def __init__(self):
        pass

    def add_gaussian_noise(self, V, noise=0.1):
        max_x, max_y = -1.0, -1.0
        min_x, min_y = sys.maxsize, sys.maxsize
        for v in V:
            max_x = max(max_x, v[0])
            max_y = max(max_y, v[1])
            min_x = min(min_x, v[0])
            min_y = min(min_y, v[1])

        size = int(np.ceil(len(V) * noise))
        for i in range(size):
            x = np.random.uniform(min_x, max_x)
            y = np.random.uniform(min_y, max_y)
            V.append((x, y))
        return V, len(V)

    def add_clustered_noise(self, V, type='', noise=0.1):
        max_x, max_y = -1.0, -1.0
        min_x, min_y = sys.maxsize, sys.maxsize
        for v in V:
            max_x = max(max_x, v[0])
            max_y = max(max_y, v[1])
            min_x = min(min_x, v[0])
            min_y = min(min_y, v[1])

        size = int(np.ceil(len(V) * noise))
        if type == '':
            NotImplementedError('No parameter given')
        elif type == 'circle':
            center = ((max_x + min_x) / 2, (max_y + min_y) / 2)
            radius = min(max_x - min_x, max_y - min_y) * 0.7 / 2
            pi = math.pi
            for i in range(size):
                V.append((math.cos(2 * pi / size * i) * radius + center[0],
                          math.sin(2 * pi / size * i) * radius + center[1]))
        elif type == 'horizontal_line':
            y_center = (max_y + min_y) / 2
            interval = (max_x - min_x) / size
            current_x = min_x
            for i in range(size):
                V.append((current_x, y_center))
                current_x += interval
        elif type == 'vertical_line':
            x_center = (max_x + min_x) / 2
            interval = (max_y - min_y) / size
            current_y = min_y
            for i in range(size):
                V.append((x_center, current_y))
                current_y += interval
        else:
            NotImplementedError('%s not implemented' % type)
        return V, len(V)

    def delete_data_area(self, V, dataset=''):
        if dataset == 'two_moons':
            NotImplementedError('%s not implemented' % dataset)
        elif dataset == 'two_circles':
            NotImplementedError('%s not implemented' % dataset)
        elif dataset == 'aniso':
            NotImplementedError('%s not implemented' % dataset)
        elif dataset == 'varied':
            NotImplementedError('%s not implemented' % dataset)
        elif dataset == 'blobs':
            NotImplementedError('%s not implemented' % dataset)
        else:
            NotImplementedError('%s not implemented' % dataset)

    def delete_random_vertices(self, adj, percentage=0.01):
        to_delete = np.ceil(len(adj) * percentage)
        for i in range(to_delete):
            vertex = np.floor(np.random.uniform(0, 1500))
            found = False
            key = -1
            for j in range(len(adj)):
                if adj[j][0] == vertex:
                    found = True
                    key = j
                    break
            if found:
                if key >= 0:
                    del adj[key]
                    for j in range(len(adj)):
                        found = False
                        for k in range(len(adj[j][1])):
                            if adj[j][1][k] == vertex:
                                found = True
                                key = k
                                break
                        if found:
                            del adj[j][1][key]
        return adj

    def delete_random_edges(self, adj, percentage=0.01):
        edges = [len(vertex[1]) for vertex in adj]
        E = sum(edges) / 2  # every edge is in there twice
        to_delete = np.ceil(E * percentage)
        for i in range(to_delete):
            u, v = np.floor(np.random.uniform(0, 1500)), np.floor(np.random.uniform(0, 1500))
            while u == v:
                v = np.floor(np.random.uniform(0, 1500))
            u_found, v_found = False, False
            key_u, key_v = (-1, -1), (-1, -1)
            for k in range(len(adj)):
                if adj[k][0] == u:
                    for j in range(len(adj[k][1])):
                        if adj[k][1][j] == v:
                            key_u = (k, j)
                            u_found = True
                if adj[k][0] == v:
                    for j in range(len(adj[k][1])):
                        if adj[k][1][j] == u:
                            key_v = (k, j)
                            v_found = True
                if u_found and v_found:
                    break
            if u_found and v_found:
                del adj[key_u[0]][1][key_u[1]]
                del adj[key_v[0]][1][key_v[1]]
            else:
                i -= 1
                print(u, v)
        return adj


