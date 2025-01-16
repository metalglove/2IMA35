import math
import abc
from pyspark import SparkContext

class GreedySubmodularV2:
    def __init__(self, sc: SparkContext, coords_x, coords_y):
        self.points = list(zip(coords_x, coords_y))
        self.sc = sc

    def run(self, k: int, circles: list[tuple[int, int, int]]):
        # we define the set cover problem oracle
        circles_dict = dict[int, tuple[int, int, int]]()
        for idx, circle in enumerate(circles):
            circles_dict[idx] = circle
        problem = SetCoverProblemOracle(self.points, circles_dict)

        def find_max_circle_mapper1(C_j):
            (max_v, max_c_id) = max([(problem.f(c_id), c_id) for c_id in C_j])
            return max_c_id, max_v
        
        def find_max_circle_reducer1(C1, C2):
            (max_a_id, max_a_v) = C1
            (max_b_id, max_b_v) = C2
            if max_a_v >= max_b_v:
                return (max_a_id, max_a_v)
            return (max_b_id, max_b_v)

        # we iterate until we have found k circles that greedily maximize the set cover.
        for i in range(k):
            print(f'round {i + 1}')

            # prepare the number of partitions (machines in the analysis) to distribute the circles to.
            n = len(problem.C.keys())
            j = math.ceil(math.sqrt(n))

            # prepare the circle ids in the number of partitions (machines).
            circle_ids = self.sc. \
                parallelize(problem.C.keys(), numSlices=j). \
                glom()

            # map and reduce from each circle subset to the maximum circle coverage
            (max_circle_id, _) = circle_ids. \
                map(find_max_circle_mapper1). \
                reduce(find_max_circle_reducer1)
            
            # we are now machine M_$, assume we are running locally.
            # update the universe by subtracting the points that are covered by the maximum circle
            # and save the maximum circle and points covered.
            problem.add(max_circle_id)

        # return the circle set cover and covered points
        return problem.S, problem.covered_points



class ProblemOracle(metaclass=abc.ABCMeta):
    def __init__(self, U: list):
        self.U = U

    def fs(self, ids) -> int:
        return sum([self.f(id) for id in ids])
    
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
    def __init__(self, P: list[tuple[int, int]], C: dict[int, tuple[int, int, int]]):
        super().__init__(P)
        self.C = C
        self.S = set[tuple[int, int, int]]()
        self.covered_points = set()

    def add(self, id):
        # find the circle in the circle set and remove it.
        (x_center, y_center, r) = self.C.pop(id)
        
        # prepare the selected points set.
        selected_points = set[tuple[int, int]]()

        # find points in universe.
        for (x, y) in self.U:
            if self.__in_circle(x_center, y_center, r, x, y):
                selected_points.add((x, y))
        
        # remove points from universe.
        for point in selected_points:
            self.U.remove(point)
        
        # store all covered points.
        self.covered_points = self.covered_points.union(selected_points)

        # add circle covering the points.
        self.S.add((x_center, y_center, r))
    
    def f(self, id) -> int:
        (x_center, y_center, r) = self.C.get(id)
        i = 0
        for (x, y) in self.U:
            if self.__in_circle(x_center, y_center, r, x, y):
                i += 1
        return i

    def __in_circle(self, center_x, center_y, radius, x, y):
        dist = math.sqrt((center_x - x) ** 2 + (center_y - y) ** 2)
        return dist <= radius