from hopfield import *
from tabulate import tabulate


def main():
    myHop = HopfieldNet()
    P_errors_ref = [0.001, 0.0036, 0.01, 0.05, 0.1]
    
    capacities = []
    for p in P_errors_ref:
        capacity = get_capacity(neurons=100, P_error_ref=p, disconnect_proportion=0.1)
        capacities.append(capacity)

    print(tabulate(capacities, headers=['P_err reached', 'Capacity']))

if __name__ == '__main__':
    main()