from support_vector_machine import SupportVectorMachine
import random


def main():
    data = setup_data()
    machine = SupportVectorMachine(data)



def setup_data():
    classA = [(random.normalvariate(-1.5, 1), random.normalvariate(0.5, 1), 1.0) for _ in range(5)] + \
             [(random.normalvariate(1.5, 1), random.normalvariate(0.5, 1), 1.0) for _ in range(5)]
    classB = [(random.normalvariate(0.0, 0.5), random.normalvariate(-0.5, 0.5), -1.0) for _ in range(10)]
    data = classA + classB
    random.shuffle(data)
    return data


if __name__ == '__main__':
    main()
