from ising import IsingModel
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.ion()


def main():
    myIsing = IsingModel(rows=4,cols=5,init_temp=20)
    myIsing.run()


if __name__ == '__main__':
    main()