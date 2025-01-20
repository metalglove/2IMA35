from matplotlib import pyplot as plt
import numpy as np
from pyspark import SparkConf, SparkContext
from algorithms.GreedySubmodularV2 import GreedySubmodularV2
from algorithms.GreedySubmodular import GreedySubmodular
from time import time
import findspark
import pandas as pd

def plot(balls, points, num_balls, size, minlim, maxlim):
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

def generate_circle(minlim, maxlim, maxradius):
    x = np.random.randint(minlim, maxlim)
    y = np.random.randint(minlim, maxlim)
    r = np.random.randint(1, maxradius)
    return (x, y, r)

def generate_circles(circles_n, minlim, maxlim):
    circless = set()
    for i in range(circles_n):
        circle = generate_circle(minlim, maxlim, maxlim / 50)
        circless.add(circle)
    return circless

def generate_dataset(size, minlim, maxlim) -> tuple[list, list]:
    coords_x = np.random.randint(minlim, maxlim, size)
    coords_y = np.random.randint(minlim, maxlim, size)
    return (list(coords_x), list(coords_y))

def augment_dataset(coordsx, coordsy, number_of_points_to_add, minlim, maxlim):
    coords_x, coords_y = generate_dataset(number_of_points_to_add, minlim, maxlim)
    return coords_x + coordsx, coords_y + coordsy

def save_experiment(name, timings):
    df = pd.DataFrame(timings)
    df.to_csv(name)

def spark_dry_run(sc):
    balls = set[tuple[int, int, int]]([(100, 300, 400), (5000, 5000, 1000), (800, 300, 500), (2000, 300, 500), (2000, 7000, 500), (7000, 2000, 1000), (3000, 4000, 1500), (1000, 8000, 3000)])
    coords_x, coords_y = generate_dataset(1000, 0, 10000)
    alg = GreedySubmodularV2(sc, coords_x, coords_y)
    _ = alg.run(3, balls)

def run_greedy_submodular(sc, coords_x, coords_y, k, circles):
    alg = GreedySubmodularV2(sc, coords_x.copy(), coords_y.copy())
    t0 = time()
    _ = alg.run(k, circles.copy())
    t1 = time()
    return t1 - t0

def experiment1(sc):
    '''
    Experiment with a constant circles set with varying positions and sizes.
    However, the size of the point set increases by 1000 points, 100 times.
    Furthermore, this is done k times.
    '''
    circles_n = 50
    minlim = 0
    maxlim = 10_000
    circless = generate_circles(circles_n, minlim, maxlim)

    timings = []

    ks = 10
    for k in range(1, ks + 1):
        for i in range(100):
            size = (i + 1) * 1000
            coords_x, coords_y = generate_dataset(size)

            timing = run_greedy_submodular(sc, coords_x, coords_y, k, circless)

            timings.append({ 'size': size, 'timing': timing, 'k': k })
            print(f'k = {k}, size = {size}, timing = {timing}')

    return timings

def experiment2(sc):
    '''
    Experiment with constant number of points but increasing number of circles
    '''
    k = 15
    points_n = 10_000
    minlim = 0
    maxlim = 10_000

    coords_x, coords_y = generate_dataset(points_n, minlim, maxlim)
    og_circless = generate_circles(k, minlim, maxlim)

    timings = []
    for j in range(10):
        circless = og_circless.copy()
        for i in range(k, 100):
            if i > k:
                circless.add(generate_circle(minlim, maxlim, maxlim / 50))

            timing = run_greedy_submodular(sc, coords_x, coords_y, k, circless)

            timings.append({ 'size': points_n, 'timing': timing, 'k': k, 'circles': len(circless) })
            print(f'[{j}] k = {k}, size = {points_n}, timing = {timing}, circles = {len(circless)}')

    return timings

def experiment3(sc):
    '''
    Experiment with constant number of points but increasing number of circles (randomized)
    '''
    k = 15
    points_n = 10_000
    minlim = 0
    maxlim = 10_000

    coords_x, coords_y = generate_dataset(points_n, minlim, maxlim)

    timings = []
    for j in range(10):
        for i in range(k, 100):
            circless = generate_circles(i, minlim, maxlim)

            timing = run_greedy_submodular(sc, coords_x, coords_y, k, circless)

            timings.append({ 'size': points_n, 'timing': timing, 'k': k, 'circles': len(circless) })
            print(f'[{j}] k = {k}, size = {points_n}, timing = {timing}, circles = {len(circless)}')

    return timings

def experiment4(sc):
    '''
    Experiment with number of points and circles increasing in a ratio of 5000:1 (lol)
    '''
    k = 15
    points_n = 10_000
    minlim = 0
    maxlim = 10_000

    og_coords_x, og_coords_y = generate_dataset(points_n, minlim, maxlim)
    og_circless = generate_circles(k, minlim, maxlim)

    timings = []
    for j in range(10):
        coords_x = og_coords_x.copy()
        coords_y = og_coords_y.copy()
        circless = og_circless.copy()
        for i in range(50):
            if i > 0:
                coords_x, coords_y = augment_dataset(coords_x, coords_y, 5000, minlim, maxlim)
                points_n = len(coords_x)
                circless.add(generate_circle(minlim, maxlim, maxlim / 50))

            timing = run_greedy_submodular(sc, coords_x, coords_y, k, circless)

            timings.append({ 'size': points_n, 'timing': timing, 'k': k, 'circles': len(circless) })
            print(f'[{j}] k = {k}, size = {points_n}, timing = {timing}, circles = {len(circless)}')

    return timings

def experiment5(sc):
    '''
    Experiment with a constant circles set with varying positions and sizes.
    However, the size of the point set increases by 1000 points, 100 times.
    Furthermore, this is done k times.
    '''
    circles_n = 50
    minlim = 0
    maxlim = 10_000
    circless_og = generate_circles(circles_n, minlim, maxlim)

    timings = []

    ks = 10
    size = 25000
    coords_x, coords_y = generate_dataset(size, minlim, maxlim)
    grow_factor = 1.2
    

    for j in range(10):
        circless = circless_og.copy()
        for i in range(1, ks + 1):
            timing = run_greedy_submodular(sc, coords_x, coords_y, ks, circless)

            timings.append({ 'grow_factor': grow_factor, 'timing': timing, 'k': ks, 'i': i, 'j': j})
            print(f'k = {ks}, size = {size}, timing = {timing}, i = {i}, j = {j}')

            circless = {(x, y, r * grow_factor) for (x, y, r) in circless}

    return timings

def main():
    timings = []
    conf = SparkConf('local[]').setAppName('GreedySubmodular').set("spark.executor.memory", "4g").set("spark.driver.memory", "2g")
    sc = SparkContext.getOrCreate(conf=conf)

    # initialze pyspark (dry run) to remove initialization overhead in performance comparisons
    spark_dry_run(sc)

    timings = experiment5(sc)
    save_experiment('experiment5.csv', timings)

if __name__ == "__main__":
    np.random.seed(5128095)
    findspark.init()
    main()
