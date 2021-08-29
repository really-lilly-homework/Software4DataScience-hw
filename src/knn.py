"""This module performs the k Nearest Neighbor regression algorithm on
    a given data set. """


import numpy


def knn_regression(n_neighbors, data, query):
    """
    Performs the k Nearest Neighbor regression algorithm.
        n_neighbors (int) - number of neighbors to average
        data (np.array) - data set
        query (np.arry) - a data point of interest
        Returns: mean_label (double) - the value of the queried data point
    """

    if isinstance(n_neighbors) != int:
        raise TypeError("n_neighbors must be an integer")
    if n_neighbors <= 0:
        raise ValueError("n_nieghbors must be positive")
    if isinstance(data) != numpy.ndarray:
        raise TypeError("data must be in the form of a numpy array")
    if isinstance(query) != numpy.ndarray:
        raise TypeError("query must be a numpy array")
    points, labels = numpy.shape(data)
    if labels-len(query) != 1:
        raise ValueError("data or query array shape is incorrect")
    if n_neighbors > points:
        raise ValueError("number of neighbors exceeds number of data points")
    distances = numpy.empty([points, 2])
    for i in range(points):
        point = data[i][0:labels-1]
        distances[i][0] = numpy.linalg.norm(query - point)
        distances[i][1] = i
    distances = distances[distances[:, 0].argsort()]
    compare_labels = numpy.empty([n_neighbors, 1], dtype=float)
    for i in range(n_neighbors):
        compare_labels[i] = data[int(distances[i][1])][2]
    mean_label = round(numpy.mean(compare_labels), 2)
    return mean_label
