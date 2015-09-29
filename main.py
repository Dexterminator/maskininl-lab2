from support_vector_machine import SupportVectorMachine


def main():
    machine = SupportVectorMachine()
    x = (1, 2, 1)
    y = (2, 3, -1)
    print(machine.linear_kernel(x, y))
    print(machine.polynomial_kernel(x, y, 2))


if __name__ == '__main__':
    main()
