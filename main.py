import random

from cvxopt.base import matrix

from support_vector_machine import SupportVectorMachine
import numpy
import pylab


def main():
    classA, classB = setup_data()
    data = classA + classB
    random.shuffle(data)
    plot(classA, classB, data, 1)
    plot(classA, classB, data, 2)
    plot(classA, classB, data, 3)
    # plot(classA, classB, data, 4)


def plot(classA, classB, data, kernel):
    machine = SupportVectorMachine(data, kernel)
    plot_decision_boundary(machine)
    plot_data_points(classA, classB)
    pylab.clf()


def setup_data():
    classA = [(random.normalvariate(-1.5, 1), random.normalvariate(0.5, 1), 1.0) for _ in range(5)] + \
             [(random.normalvariate(1.5, 1), random.normalvariate(0.5, 1), 1.0) for _ in range(5)]
    classB = [(random.normalvariate(0.0, 0.5), random.normalvariate(-0.5, 0.5), -1.0) for _ in range(10)]
    return classA, classB


def plot_decision_boundary(machine):
    x_range = numpy.arange(-4, 4, 0.05)
    y_range = numpy.arange(-4, 4, 0.05)
    grid = matrix([[machine.indicator((x, y)) for y in y_range] for x in x_range])
    pylab.contour(x_range, y_range, grid,
                  (-1.0, 0.0, 1.0),
                  colors=('red', 'black', 'blue'),
                  linewidths=(1, 3, 1))


def plot_data_points(classA, classB):
    pylab.hold(True)
    pylab.plot([p[0] for p in classA],
               [p[1] for p in classA],
               'bo')
    pylab.plot([p[0] for p in classB],
               [p[1] for p in classB],
               'ro')
    pylab.show()


if __name__ == '__main__':
    main()
