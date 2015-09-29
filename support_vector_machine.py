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

    def radial_basis_kernel(self):
        pass

    def sigmoid_kernel(self):
        pass

    def build_p_matrix(self, data_points):
        pass
