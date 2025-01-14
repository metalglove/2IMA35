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
        # we define the oracle
        problem = SetCoverProblemOracle(self.points, balls)

        def find_max_ball_mapper1(C_j):
            (max_v, max_c_id) = max([(problem.f(c_id), c_id) for c_id in C_j])
            return max_c_id, max_v
        
        def find_max_ball_reducer1(C1, C2):
            (max_a_id, max_a_v) = C1
            (max_b_id, max_b_v) = C2
            if max_a_v >= max_b_v:
                return (max_a_id, max_a_v)
            return (max_b_id, max_b_v)

        for i in range(k):
            n = len(problem.C)
            j = math.ceil(math.sqrt(n))

            # prepare the ball ids in the number of machines (partitions)
            ball_ids = self.sc.range(start=0, end=n, numSlices=j).glom()

            # map and reduce from each ball subset to the maximum ball
            max_ball_id = ball_ids.map(find_max_ball_mapper1).reduce(find_max_ball_reducer1)
            
            # we are now machine M_$, assume we are running locally.
            problem.add(max_ball_id)

        return problem.S, problem.covered_points

import abc

class ProblemOracle(metaclass=abc.ABCMeta):
    def __init__(self, U: list):
        self.__U = U

    def get_universe(self) -> list:
        return self.__U

    @abc.abstractmethod
    def fs(self, ids) -> int:
        pass
    
    @abc.abstractmethod
    def f(self, id) -> int:
        pass

    # for Circle cover
    # contains the set of points (P)
    # contains the set of circles (C)
    # has a function that returns a list of elements , where each element is mapped to a specific circle in C
    # -elements can just be simple integer id's
    # 
    # function f() (the oracle function)
    # -input: the list of elements
    # -output: in our case, the number of points covered by the union of circles corresponding to the id's
    # 
    # function get_universe() returns the elements (aka the domain of the oracle function)
    # -no input
    # -output: in our case, a list of representations of the circles

class SetCoverProblemOracle(ProblemOracle):
    def __init__(self, P: list[tuple[int, int]], C: list[tuple[int, int, int]]):
        super().__init__(P)
        self.C = C
        self.S = set()
        self.covered_points = set()

    def add(self, id):
        (x_center, y_center, r) = self.C[id]
        selected_points = set[tuple[int, int]]()

        # find points in universe
        for (x, y) in self.get_universe():
            if self.__in_circle(x_center, y_center, r, x, y):
                selected_points.add((x, y))
        
        # remove points from universe
        for point in selected_points:
            self.get_universe().remove(point)
        
        # store all covered points
        self.covered_points = self.covered_points.union(selected_points)

        # add ball covering the points
        self.S.add((x_center, y_center, r))

        # remove from C
        del self.C[id]

    def fs(self, ids) -> int:
        return sum([self.f(id) for id in ids])
    
    def f(self, id) -> int:
        (x_center, y_center, r) = self.C[id]
        i = 0
        for (x, y) in self.get_universe():
            if self.__in_circle(x_center, y_center, r, x, y):
                i += 1
        return i

    def __in_circle(self, center_x, center_y, radius, x, y):
        dist = math.sqrt((center_x - x) ** 2 + (center_y - y) ** 2)
        return dist <= radius