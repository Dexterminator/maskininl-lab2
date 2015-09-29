from cvxopt.solvers import qp
from cvxopt.base import matrix
import numpy, pylab, random, math


class SupportVectorMachine:
    def __init__(self):
        self.alphas = []

    @staticmethod
    def linear_kernel(x, y):
        return numpy.transpose(x).dot(y) + 1

    @staticmethod
    def polynomial_kernel(x, y, p):
        return math.pow(numpy.transpose(x).dot(y) + 1, p)

    @staticmethod
    def radial_basis_kernel(self):
        pass

    @staticmethod
    def dim2(data_point):
        return data_point[0:2]

    def sigmoid_kernel(self):
        pass

    @staticmethod
    def build_p_matrix(data_points):
        p_matrix = numpy.empty([len(data_points), len(data_points)])
        for i, x_i in enumerate(data_points):
            for j, x_j in enumerate(data_points):
                p_matrix[i][j] = x_i[2] * x_j[2] * SupportVectorMachine.linear_kernel(x_i[0:2], x_j[0:2])
        return p_matrix
