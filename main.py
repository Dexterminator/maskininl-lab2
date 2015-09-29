from support_vector_machine import SupportVectorMachine
import random


def main():
    machine = SupportVectorMachine()
    x = (1, 2, 1)
    y = (2, 3, -1)
    print(machine.linear_kernel(x, y))
    print(machine.polynomial_kernel(x, y, 2))
    print(SupportVectorMachine.dim2(x))
    classA = [(random.normalvariate(-1.5, 1), random.normalvariate(0.5, 1), 1.0) for _ in range(5)] + \
             [(random.normalvariate(1.5, 1), random.normalvariate(0.5, 1), 1.0) for _ in range(5)]
    classB = [(random.normalvariate(0.0, 0.5), random.normalvariate(-0.5, 0.5), -1.0) for _ in range(10)]
    data = classA + classB
    random.shuffle(data)
    p_matrix = machine.build_p_matrix(data)
    print(p_matrix)


if __name__ == '__main__':
    main()
