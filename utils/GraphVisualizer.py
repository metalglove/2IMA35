
from matplotlib import pyplot as plt


class GraphVisualizer:
    def __init__(self, G):
        self.G = G
        self.plot_calls = 0

    def plot_graph(self, title = ''):
        plt.figure()

        xs, ys = [], []
        for x in self.G.V.keys():
            xs.append(self.G.points[x][0])
            ys.append(self.G.points[x][1])
        plt.title(title)
        plt.scatter(xs, ys, c='r')

        ax = plt.gca()
        if self.plot_calls == 0:
            self.xlim = ax.get_xlim()
            self.ylim = ax.get_ylim()
        else:
            ax.set_xlim(self.xlim)
            ax.set_ylim(self.ylim)

        self.plot_calls = self.plot_calls + 1
