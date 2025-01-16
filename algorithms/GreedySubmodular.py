# max coverage
import math

from pyspark import SparkContext

def in_circle(center_x, center_y, radius, x, y):
    dist = math.sqrt((center_x - x) ** 2 + (center_y - y) ** 2)
    return dist <= radius

class GreedySubmodular:
    def __init__(self, sc: SparkContext, coords_x, coords_y):
        self.points = list(zip(coords_x, coords_y))
        self.sc = sc

    def run(self, k: int, circles: list[tuple[int, int, int]]):
        # distribute points equally among machines V_j = { e_{j\sqrt{n}}, e_{(j\sqrt{n}) + 1}, \cdots, e_{(j\sqrt{n}) + \sqrt{n}} }
        n = len(self.points)
        j = math.ceil(math.sqrt(n))

        def f(V_j, circle) -> int:
            (x_center, y_center, r) = circle
            i = 0
            for (x, y) in V_j:
                if in_circle(x_center, y_center, r, x, y):# and (x, y) not in O_points:
                    i += 1
            return i

        def f2(V_j, circle) -> int:
            (x_center, y_center, r) = circle
            selected_points = set[tuple[int, int]]()
            for (x, y) in V_j:
                if in_circle(x_center, y_center, r, x, y):# and (x, y) not in O_points:
                    selected_points.add((x, y))
            return len(selected_points), selected_points
                
        def find_max_circle_map(V_j, circles) -> tuple[tuple[int, int, int], int]:
            # find the maximum value for f(circle, V_j) in circles
            # print(f'--{j}--')
            (value, max_circle) = max((f2(V_j, ball), ball) for ball in circles)
            (max_circle_coverage_value, points) = value
            #print(f'max_circle: {max_circle}, coverage: {points}')
            # print(f'M_j: {j}, max_circle: {max_circle}, coverage: {max_circle_coverage_value}')
            return max_circle, value
        
        def find_max_circle_reduce(a, b):
            (aa, ab) = a
            (ba, bb) = b
            return (aa + ba, ab | bb)

        # find element e whose result is maximum (max_circle, max_circle_coverage_value) 
            # .glom() \
            # .combineByKey(
            #     lambda c: c,
            #     lambda C, v: C + v,
            #     lambda C1, C2: C1 + C2
            # ) \
        # max_circles = self.sc.parallelize(self.points, j) \
        #     .mapPartitionsWithIndex(lambda j, V_j: find_max_circle_map(j, V_j, circles)) \
        #     .reduceByKey(find_max_circle_reduce) \
        #     .collect()
        
        def simple_reduce(a, b):
            (a_circle, a_v) = a
            (a_max_circle_coverage_value, points) = a_v
            (b_circle, b_v) = b
            (b_max_circle_coverage_value, points) = b_v
            if a_max_circle_coverage_value >= b_max_circle_coverage_value:
                return a
            return b

        O_circles = set[tuple[int, int, int]]()
        O_points = set[tuple[int, int]]()

        for i in range(k):
            # map the points V_j to each partition (machine), find the maximum circle in each machine,
            # reduce by summing the maximum circle and then finding the true maximum circle.
            max_circle = self.sc.parallelize(self.points, j) \
                .glom() \
                .map(lambda V_j: find_max_circle_map(V_j, circles)) \
                .reduceByKey(find_max_circle_reduce) \
                .map(lambda c: c) \
                .reduce(simple_reduce)
        
            (circle, (val, points)) = max_circle
            # remove from V
            for p in points:
                self.points.remove(p)
            circles.remove(circle)

            # add to S
            O_circles.add(circle)
            O_points = O_points.union(points)

        return O_circles, O_points
