import math

from cvxopt.base import matrix

from cvxopt.solvers import qp
import numpy

from functools import partial


class SupportVectorMachine:
    def __init__(self, data_points, kernel_id):
        kernels = {1: self.linear_kernel, 2: partial(self.polynomial_kernel, p=2),
                   3: partial(self.radial_basis_kernel, sigma=2), 4: partial(self.sigmoid_kernel, k=7, delta=-4)}
        self.kernel = kernels[kernel_id]
        # self.kernel = partial(self.polynomial_kernel, p=2)
        # self.kernel = partial(self.radial_basis_kernel, sigma=2)
        # self.kernel = partial(self.sigmoid_kernel, k=7, delta=-4)
        self.p = self.build_p_matrix(data_points)
        self.q = self.build_q_vector(len(data_points))
        self.h = self.build_h_vector(len(data_points))
        self.g = self.build_g_matrix(len(data_points))
        alphas = self.call_qp()
        self.alpha_class = self.without_zeroes(zip(alphas, data_points))
        print(self.alpha_class)

    def linear_kernel(self, x, y):
        return numpy.transpose(x).dot(y) + 1

    def polynomial_kernel(self, x, y, p):
        return math.pow(numpy.transpose(x).dot(y) + 1, p)

    def radial_basis_kernel(self, x, y, sigma):
        n = numpy.linalg.norm(numpy.array(x) - numpy.array(y))
        return math.exp((-1) * n * n / (2 * sigma * sigma))

    def sigmoid_kernel(self, x, y, k, delta):
        return numpy.tanh(k * numpy.transpose(x).dot(y) - delta)

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
                p_matrix[i][j] = x_i[2] * x_j[2] * self.kernel(x_i[0:2], x_j[0:2])
        return p_matrix

    def call_qp(self):
        r = qp(matrix(self.p), matrix(self.q), matrix(self.g), matrix(self.h))
        return list(r['x'])

    def without_zeroes(self, alpha_classes):
        return [(a, data_point) for a, data_point in alpha_classes if abs(a) > 1e-5]

    def indicator(self, new_data_point):
        indicator_sum = 0
        for a, data_point in self.alpha_class:
            indicator_sum += a * data_point[2] * self.kernel(new_data_point[0:2], data_point[0:2])
        return indicator_sum
