from matplotlib import pyplot as plt
import numpy as np
from algorithms.GreedySubmodular import GreedySubmodular

def main():
    # k
    k = 2

    # x, y, r
    balls = set[tuple[int, int, int]]([(1, 3, 4), (50, 50, 10), (8, 3, 5), (20, 3, 5), (20, 70, 5)])
    num_balls = len(balls)

    # points
    num_points = 400
    minlim = 0
    maxlim = 100
    coords_x = np.random.randint(minlim, maxlim, num_points)
    coords_y = np.random.randint(minlim, maxlim, num_points)

    def plot(balls, points):
        colors = plt.get_cmap('viridis')(np.linspace(0, 1, len(balls)))
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.set_title(f'Max cover set k = {k}, balls = {num_balls}, n = {num_points},')
        plt.xlim(minlim, maxlim)
        plt.ylim(minlim, maxlim)

        for i, (x, y, r) in enumerate(balls):
            circle = plt.Circle((x, y), r, color=colors[i])
            ax.add_patch(circle)

        x, y = zip(*points)
        ax.scatter(x, y)
        plt.show(block = False)

    plot(balls, list(zip(coords_x, coords_y)))

    # algorithm
    alg = GreedySubmodular(coords_x, coords_y)
    O_balls, O_points = alg.run(k, balls)

    plot(O_balls, O_points)

    plt.show(block = True)

if __name__ == "__main__":
    np.random.seed(56)
    main()