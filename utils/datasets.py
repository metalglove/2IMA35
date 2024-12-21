from sklearn.datasets import make_blobs, make_circles, make_moons


points1a = [(3, 1), (4, 2),
            (5, 3), (6, 3), (7, 3),
            (5, 4), (6, 4), (7, 4), (4, 4), (3, 4), (2, 4),
            (5, 5), (6, 5), (7, 5),
            (3, 7), (4, 6),

            (1,13), (1,14), (2,13), (2,14), (3,13), (3,14), (4,13), (4,14), (5,13), (5,14), (6,13), (6,14),
            (5,15), (6,15), (5,12), (6,12),(5,11), (6,11), (5,11),

            (10,10), (10,11), (11,9),(12,9),(13,9),(14,9),(15,10),(15,11),(11,12),(12,12),(13,12),(14,12)]

points1b = [(1, 13), (1, 14), (2, 13), (2, 14), (3, 13), (3, 14), (4, 13), (4, 14),
          (3, 4), (4, 4), (5, 4),
           (5, 6), (6, 6), (7, 6),
            (8, 8), (9, 8), (10, 8),
             (10, 10), (11, 10), (12, 10),
              (13, 1), (14, 1), (15, 2), (15, 3), (13, 4), (14, 4)]


points1c = [(1, 15), (1, 14), (1, 13), (1, 12), (1, 11), (1, 10), (1, 9), (1, 8), (1, 7),
            (2, 15), (2, 14), (2, 13), (2, 12), (2, 11), (2, 10), (2, 9), (2, 8), (2, 7),
            (3, 7), (4, 7), (5, 7), (6, 7),

            (7, 15), (9, 15), (11, 15), (13, 15), (15, 15),
            (7, 13), (9, 13), (11, 13), (13, 13), (15, 13),

            (8, 1), (10, 1),
            (9, 2), (11, 2),
            (8, 3), (10, 3), (12, 3),
            (9, 4), (11, 4), (13, 4),
            (10, 5), (12, 5), (14, 5),
            (11, 6), (13, 6), (15, 6),
            (12, 7), (14, 7),
            (13, 8), (15, 8)]


def __generate_maccie():
    import pandas as pd
    df = pd.read_csv("datasets/mcdonalds.csv",encoding='latin1', names=["lat", "long", "type", "location"])

    points = df[["lat", "long"]].values

    # value 1: x bottom-left,
    # value 2: y bottom-left,
    # value 3: x top-right,
    # value 4: y top-right
    netherlandsRect = (3.31497114423, 50.803721015, 7.09205325687, 53.5104033474)
    netherlandsRectAlt = (3.273926, 50.680777, 7.316895, 53.572979)

    def rectContains(rect, pt):
        logic = rect[0] <= pt[0] and rect[1] <= pt[1] and rect[2] >= pt[0] and rect[3] >= pt[1]
        return logic

    def filterPoints(points):
        newPoints = []

        for point in points:
            if rectContains(netherlandsRect, point):
                newPoints.append(point)

        return newPoints

    return filterPoints(points)

def generate_points(name, n_samples = 300, n_clusters = 3):
    if name == 'ans1c':
        return points1c
    elif name == 'ans1b':
        return points1b
    elif name =='ans1a':
        return points1a
    elif name == 'blobs':
        V, _ = make_blobs(n_samples = n_samples, centers = n_clusters, n_features = 2, random_state = 56)
        return list(map(tuple, V))
    elif name == 'moons':
        V, _ = make_moons(n_samples = n_samples, random_state = 56)
        return list(map(tuple, V))
    elif name == 'circles':
        V, _ = make_circles(n_samples = n_samples, factor = 0.5, random_state = 56)
        return list(map(tuple, V))
    elif name == 'maccie':
        return __generate_maccie()
    else:
        raise ValueError(f"No dataset found for name: {name}")
    