import random

from cvxopt.base import matrix

from support_vector_machine import SupportVectorMachine
import numpy
import pylab


def main():
    class_a, class_b = setup_data()
    data = class_a + class_b
    random.shuffle(data)
    s1 = 'linear'
    pylab.figure(s1)
    plot(class_a, class_b, data, 1)
    save_figure(s1)

    s2 = 'polynomial, p=2'
    pylab.figure(s2)
    plot(class_a, class_b, data, 2)
    save_figure(s2)

    s3 = 'polynomial, p=3'
    pylab.figure(s3)
    plot(class_a, class_b, data, 3)
    save_figure(s3)

    s4 = 'polynomial, p=4'
    pylab.figure(s4)
    plot(class_a, class_b, data, 4)
    save_figure(s4)

    s5 = 'radial basis, sigma=1'
    pylab.figure(s5)
    plot(class_a, class_b, data, 5)
    save_figure(s5)

    s6 = 'radial basis, sigma=2'
    pylab.figure(s6)
    plot(class_a, class_b, data, 6)
    save_figure(s6)

    s7 = 'radial basis, sigma=3'
    pylab.figure(s7)
    plot(class_a, class_b, data, 7)
    save_figure(s7)
    pylab.show()


def save_figure(s1):
    pylab.savefig(s1 + '.png')


def plot(class_a, class_b, data, kernel):
    machine = SupportVectorMachine(data, kernel)
    plot_decision_boundary(machine)
    plot_data_points(class_a, class_b)


def setup_data():
    class_a = [(random.normalvariate(-1.5, 1), random.normalvariate(0.5, 1), 1.0) for _ in range(5)] + \
              [(random.normalvariate(1.5, 1), random.normalvariate(0.5, 1), 1.0) for _ in range(5)]
    class_b = [(random.normalvariate(0.0, 0.5), random.normalvariate(-0.5, 0.5), -1.0) for _ in range(10)]
    return class_a, class_b


def plot_decision_boundary(machine):
    x_range = numpy.arange(-4, 4, 0.05)
    y_range = numpy.arange(-4, 4, 0.05)
    grid = matrix([[machine.indicator((x, y)) for y in y_range] for x in x_range])
    pylab.contour(x_range, y_range, grid,
                  (-1.0, 0.0, 1.0),
                  colors=('red', 'black', 'blue'),
                  linewidths=(1, 3, 1))


def plot_data_points(class_a, class_b):
    pylab.hold(True)
    pylab.plot([p[0] for p in class_a],
               [p[1] for p in class_a],
               'bo')
    pylab.plot([p[0] for p in class_b],
               [p[1] for p in class_b],
               'ro')


if __name__ == '__main__':
    main()
