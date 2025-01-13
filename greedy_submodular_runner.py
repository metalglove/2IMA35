from matplotlib import pyplot as plt
import numpy as np
from pyspark import SparkConf, SparkContext
from algorithms.GreedySubmodular import GreedySubmodular
from time import time
import findspark

def main():
    # k
    k = 2
    minlim = 0
    maxlim = 100

    def generate_dataset(size):
        # points
        coords_x = np.random.randint(minlim, maxlim, size)
        coords_y = np.random.randint(minlim, maxlim, size)
        return coords_x, coords_y

    def plot(balls, points):
        colors = plt.get_cmap('viridis')(np.linspace(0, 1, len(balls)))
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.set_title(f'Max cover set k = {k}, balls = {num_balls}, n = {size},')
        plt.xlim(minlim, maxlim)
        plt.ylim(minlim, maxlim)

        for i, (x, y, r) in enumerate(balls):
            circle = plt.Circle((x, y), r, color=colors[i])
            ax.add_patch(circle)

        x, y = zip(*points)
        ax.scatter(x, y)
        plt.show(block = False)

    # x, y, r
    balls = set[tuple[int, int, int]]([(1, 3, 4), (50, 50, 10), (8, 3, 5), (20, 3, 5), (20, 70, 5)])
    num_balls = len(balls)

    timings = []
    conf = SparkConf('local[]').setAppName('GreedySubmodular')
    sc = SparkContext.getOrCreate(conf=conf)

    for i in range(2):
        i = i + 1
        size = i * 100 
        coords_x, coords_y = generate_dataset(size)
        plot(balls, list(zip(coords_x, coords_y)))

        # algorithm
        alg = GreedySubmodular(sc, coords_x, coords_y)
        t0 = time()
        O_balls, O_points = alg.run(k, balls)
        t1 = time()
        timings.append(t1 - t0)
        print(f'timing = {t1 - t0}')

        plot(O_balls, O_points)

    # plot performance graph
    fig = plt.figure(figsize=(5, 5))
    
    plt.plot(timings)

    # show all figs
    plt.show(block = True)


if __name__ == "__main__":
    np.random.seed(56)
    findspark.init()
    main()
    