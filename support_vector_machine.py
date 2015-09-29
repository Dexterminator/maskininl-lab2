from cvxopt.solvers import qp
from cvxopt.base import matrix
import numpy, pylab, random, math


class SupportVectorMachine:
    def __init__(self, data_points):
        self.data_points = data_points
        self.p = self.build_p_matrix(data_points)
        self.q = self.build_q_vector(len(data_points))
        self.h = self.build_h_vector(len(data_points))
        self.g = self.build_g_matrix(len(data_points))
        self.alphas = self.call_qp()

    def linear_kernel(self, x, y):
        return numpy.transpose(x).dot(y) + 1

    def polynomial_kernel(self, x, y, p):
        return math.pow(numpy.transpose(x).dot(y) + 1, p)

    def radial_basis_kernel(self):
        pass

    def sigmoid_kernel(self):
        pass

    def build_q_vector(self, n):
        q = numpy.empty(n)
        q.fill(-1)
        return q

    def build_h_vector(self, n):
        return numpy.zeros(n)

    def build_g_matrix(self, n):
        g_matrix = numpy.zeros([n, n])
        for i in range(n):
            g_matrix[i][i] = -1
        return g_matrix

    def build_p_matrix(self, data_points):
        p_matrix = numpy.empty([len(data_points), len(data_points)])
        for i, x_i in enumerate(data_points):
            for j, x_j in enumerate(data_points):
                p_matrix[i][j] = x_i[2] * x_j[2] * self.linear_kernel(x_i[0:2], x_j[0:2])
        return p_matrix

    def call_qp(self):
        r = qp(matrix(self.p), matrix(self.q), matrix(self.g), matrix(self.h))
        return list(r['x'])

    def without_zeroes(self, alphas):
        return [a for a in alphas if abs(a) < 1e-5]
