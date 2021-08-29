"""This module performs unit tests on the k nearest neighbor module."""


import unittest
import numpy
from src import knn


class UnitTests(unittest.TestCase):
    """This class contains various unit tests for the knn module. """

    def test_smoke(self):
        """Smoke test for general functionality."""
        n_neighbors = 3
        data = numpy.array([[3, 1, 230],
                            [6, 2, 745],
                            [6, 6, 1080],
                            [4, 3, 495],
                            [2, 5, 260]])
        query = numpy.array([5, 4])
        result = knn.knn_regression(n_neighbors, data, query)
        self.assertTrue(isinstance(result) == int or isinstance(result) == float)

    def test_1shot_1(self):
        """One-shot test to ensure output is correctly calculated and returned."""
        n_neighbors = 3
        data = numpy.array([[3, 1, 230],
                            [6, 2, 745],
                            [6, 6, 1080],
                            [4, 3, 495],
                            [2, 5, 260]])
        query = numpy.array([5, 4])
        self.assertEqual(knn.knn_regression(n_neighbors, data, query), 773.33)

    def test_1shot_2(self):
        """Second one-shot test to ensure output is correctly calculated and returned."""
        n_neighbors = 4
        data = numpy.array([[13, -10, 100, 3.5],
                            [14, -50, 120, 4.3],
                            [56, -101, 200, 5.6],
                            [-3, -200, 15, -1.6],
                            [-50, -200, 400, -12],
                            [100, -20, 450, -4.3]])
        query = numpy.array([75, -4, 150])
        self.assertEqual(knn.knn_regression(n_neighbors, data, query), 108.75)

    def test_n_neighbors_type(self):
        """Tests that a TypeError is thrown if n_neighbors is not an integer."""
        n_neighbors = 3.4
        data = numpy.array([[3, 1, 230],
                            [6, 2, 745],
                            [6, 6, 1080],
                            [4, 3, 495],
                            [2, 5, 260]])
        query = numpy.array([5, 4])
        with self.assertRaises(TypeError):
            knn.knn_regression(n_neighbors, data, query)


    def test_n_nieghbors_value(self):
        """Tests that a ValueError is thrown if n_neighbors is negative."""
        n_neighbors = -2
        data = numpy.array([[3, 1, 230],
                            [6, 2, 745],
                            [6, 6, 1080],
                            [4, 3, 495],
                            [2, 5, 260]])
        query = numpy.array([5, 4])
        with self.assertRaises(ValueError):
            knn.knn_regression(n_neighbors, data, query)

    def test_data_type(self):
        """Tests that a TypeError is thrown if data is not an array."""
        n_neighbors = 3
        data = "String"
        query = numpy.array([5, 4])
        with self.assertRaises(TypeError):
            knn.knn_regression(n_neighbors, data, query)

    def test_data_dims1(self):
        """Tests that a ValueError is thrown if query is incorrect size."""
        n_neighbors = 3
        data = numpy.array([[3, 1, 230],
                            [6, 2, 745],
                            [6, 6, 1080],
                            [4, 3, 495],
                            [2, 5, 260]])
        query = numpy.array([5, 4, 100])
        with self.assertRaises(ValueError):
            knn.knn_regression(n_neighbors, data, query)

    def test_data_dims2(self):
        """Tests that a ValueError is thrown if the query is the incorrect length."""
        n_neighbors = 3
        data = numpy.array([[3, 1, 230],
                            [6, 2, 745],
                            [6, 6, 1080],
                            [4, 3, 495],
                            [2, 5, 260]])
        query = numpy.array([5])
        with self.assertRaises(ValueError):
            knn.knn_regression(n_neighbors, data, query)

    def test_query_type(self):
        """Tests that a TypeError is thrown if the query is the incorrect type."""
        n_neighbors = 3
        data = numpy.array([[3, 1, 230],
                            [6, 2, 745],
                            [6, 6, 1080],
                            [4, 3, 495],
                            [2, 5, 260]])
        query = (1, 2)
        with self.assertRaises(TypeError):
            knn.knn_regression(n_neighbors, data, query)

    def test_data_point_num(self):
        """Tests that a ValueError is thrown if the n_neighbors is greater than
            the number of data points."""
        n_neighbors = 8
        data = numpy.array([[3, 1, 230],
                            [6, 2, 745],
                            [6, 6, 1080],
                            [4, 3, 495],
                            [2, 5, 260]])
        query = numpy.array([5, 4])
        with self.assertRaises(ValueError):
            knn.knn_regression(n_neighbors, data, query)

if __name__ == '__main__':
    unittest.main()

SUITE = unittest.TestLoader().loadTestsFromTestCase(UnitTests)
_ = unittest.TextTestRunner().run(SUITE)
