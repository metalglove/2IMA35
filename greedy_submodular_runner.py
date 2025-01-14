from matplotlib import pyplot as plt
import numpy as np
from pyspark import SparkConf, SparkContext
from algorithms.GreedySubmodular import GreedySubmodular
from time import time
import findspark
import pandas as pd


def main():
    # k
    k = 3
    minlim = 0
    maxlim = 10_000

    def generate_dataset(size):
        # points
        coords_x = np.random.randint(minlim, maxlim, size)
        coords_y = np.random.randint(minlim, maxlim, size)
        return coords_x, coords_y

    def plot(balls, points, num_balls, size):
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

    timings = []
    # sizes = []
    conf = SparkConf('local[]').setAppName('GreedySubmodular').set("spark.executor.memory", "4g").set("spark.driver.memory", "2g")
    sc = SparkContext.getOrCreate(conf=conf)

    # initialze pyspark (dry run)
    balls = set[tuple[int, int, int]]([(100, 300, 400), (5000, 5000, 1000), (800, 300, 500), (2000, 300, 500), (2000, 7000, 500), (7000, 2000, 1000), (3000, 4000, 1500), (1000, 8000, 3000)])
    coords_x, coords_y = generate_dataset(100)
    alg = GreedySubmodular(sc, coords_x, coords_y)
    O_balls, O_points = alg.run(k, balls)

    balls_n = 50
    ballss = set()
    for i in range(balls_n):
        x = np.random.randint(minlim, maxlim)
        y = np.random.randint(minlim, maxlim)
        r = np.random.randint(1, maxlim / 50)
        balls.add((x, y, r))

    for i in range(100):
        # x, y, r
        balls = ballss.copy()
        num_balls = len(balls)

        i = i + 1
        size = i * 1000 
        coords_x, coords_y = generate_dataset(size)
        # plot(balls, list(zip(coords_x, coords_y)), num_balls, size)

        # algorithm
        alg = GreedySubmodular(sc, coords_x, coords_y)
        t0 = time()
        O_balls, O_points = alg.run(k, balls)
        t1 = time()
        timings.append({ 'size': size, 'timing': t1 - t0 })
        print(f'size = {size}, timing = {t1 - t0}')

        # plot(O_balls, O_points, num_balls, size)
    df = pd.DataFrame(timings)
    df.to_csv('timings4.csv')

    # plot performance graph
    fig = plt.figure(figsize=(5, 5))
    plt.title(f'Running time vs data set size')
    plt.plot(df['size'], df['timing'])
    plt.xlabel('size')
    plt.ylabel('seconds')
    fig.savefig('performance.png')

    # show all figs
    plt.show(block = True)


if __name__ == "__main__":
    np.random.seed(56)
    findspark.init()
    main()
    