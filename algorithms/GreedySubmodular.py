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

    def run(self, k: int, balls: list[tuple[int, int, int]]):
        # distribute points equally among machines V_j = { e_{j\sqrt{n}}, e_{(j\sqrt{n}) + 1}, \cdots, e_{(j\sqrt{n}) + \sqrt{n}} }
        n = len(self.points)
        j = math.ceil(math.sqrt(n))

        def f(V_j, ball) -> int:
            (x_center, y_center, r) = ball
            i = 0
            for (x, y) in V_j:
                if in_circle(x_center, y_center, r, x, y):# and (x, y) not in O_points:
                    i += 1
            return i

        def f2(V_j, ball) -> int:
            (x_center, y_center, r) = ball
            selected_points = set[tuple[int, int]]()
            for (x, y) in V_j:
                if in_circle(x_center, y_center, r, x, y):# and (x, y) not in O_points:
                    selected_points.add((x, y))
            return len(selected_points), selected_points
                
        def find_max_ball_map(V_j, balls) -> tuple[tuple[int, int, int], int]:
            # find the maximum value for f(ball, V_j) in balls
            # print(f'--{j}--')
            (value, max_ball) = max((f2(V_j, ball), ball) for ball in balls)
            (max_ball_coverage_value, points) = value
            #print(f'max_ball: {max_ball}, coverage: {points}')
            # print(f'M_j: {j}, max_ball: {max_ball}, coverage: {max_ball_coverage_value}')
            return max_ball, value
        
        def find_max_ball_reduce(a, b):
            (aa, ab) = a
            (ba, bb) = b
            return (aa + ba, ab | bb)

        # find element e whose result is maximum (max_ball, max_ball_coverage_value) 
            # .glom() \
            # .combineByKey(
            #     lambda c: c,
            #     lambda C, v: C + v,
            #     lambda C1, C2: C1 + C2
            # ) \
        # max_balls = self.sc.parallelize(self.points, j) \
        #     .mapPartitionsWithIndex(lambda j, V_j: find_max_ball_map(j, V_j, balls)) \
        #     .reduceByKey(find_max_ball_reduce) \
        #     .collect()
        
        def simple_reduce(a, b):
            (a_ball, a_v) = a
            (a_max_ball_coverage_value, points) = a_v
            (b_ball, b_v) = b
            (b_max_ball_coverage_value, points) = b_v
            if a_max_ball_coverage_value >= b_max_ball_coverage_value:
                return a
            return b

        O_balls = set[tuple[int, int, int]]()
        O_points = set[tuple[int, int]]()

        for i in range(k):
            max_ball = self.sc.parallelize(self.points, j) \
                .glom() \
                .map(lambda V_j: find_max_ball_map(V_j, balls)) \
                .reduceByKey(find_max_ball_reduce) \
                .map(lambda c: c) \
                .reduce(simple_reduce)
        
            #print(f'final max_ball: {max_ball}')
            (ball, (val, points)) = max_ball
            # remove from V
            for p in points:
                self.points.remove(p)
            balls.remove(ball)

            # add to S
            O_balls.add(ball)
            O_points = O_points.union(points)

        return O_balls, O_points
