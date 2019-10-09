import numpy as np
from utils import sgn
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s") 
log = logging.getLogger(__name__)

TEMPERATURE_DECREASING_FACTOR = 0.95
MAX_ITERATIONS_PER_TEMPERATURE = 2048

class IsingModel():

    def __init__(self, rows, cols, init_temp):
        self.rows = rows
        self.cols = cols
        self.lattice = np.vectorize(sgn)( np.random.rand(self.rows,self.cols), 0.5)
        self.temperature = init_temp
        self.iterations_per_temp_count = 0
        self.magnetization = self.lattice.sum()

    def isNotMagnetized(self):
        if (abs(self.magnetization) == self.cols*self.rows):
            return False
        else:
            return True

    def _accessLattice(self,indexes):
        row_index,col_index = indexes
        
        if (row_index >= self.rows) or (row_index < 0):
            return 0
        elif (col_index >= self.cols) or (col_index < 0):
            return 0
        else:
            return self.lattice[row_index,col_index]

    # Magnetization is simply the sum of all components in lattice
    def getMagnetization(self):
        return self.lattice.sum()

    def _updateTemperature(self):
        if self.iterations_per_temp_count == MAX_ITERATIONS_PER_TEMPERATURE:
            self.temperature *= TEMPERATURE_DECREASING_FACTOR
            self.iterations_per_temp_count = 1
        else:
            self.iterations_per_temp_count += 1


    def _getLatticeWithFlippedCandidateAt(self, index):
        new_lattice = np.copy(self.lattice)
        new_lattice[index] *= -1
        return new_lattice

    def _getCellEnergy(self, index, lattice):
        i,j = index
        return -1/2*self.lattice[i,j]*(
                                  self._accessLattice((i+1,j))
                                + self._accessLattice((i-1,j))
                                + self._accessLattice((i,j+1)) 
                                + self._accessLattice((i,j-1))
                                )

    def _getEnergy(self,lattice):
        energy = 0
        for i in range(self.rows):
            for j in range(self.cols):
                energy = energy + self._getCellEnergy((i,j),lattice)
        return energy

    def _acceptanceProb(self, dE):
        try:
            probability = np.exp(-np.true_divide(dE,self.temperature))
            return probability
        except ZeroDivisionError:
            log.warning("Catching ZeroDivisionError...")
            exit()

    def _acceptWithProb(self, delta_energy):
        if (np.random.random() <= self._acceptanceProb(delta_energy) ): # random() returns value between 0 and 1
            return True
        else:
            return False

    def _updateLattice(self, index, delta_energy):
        if delta_energy < 0:
            self.lattice[index] *= -1
        elif self._acceptWithProb(delta_energy): 
            self.lattice[index] *= -1
       
    def _iterate(self):
        # Choose random element in lattice and get it flipped
        index = np.random.randint(self.rows), np.random.randint(self.cols)
        new_lattice = self._getLatticeWithFlippedCandidateAt(index)

        old_E = self._getEnergy(self.lattice)
        new_E = self._getEnergy(new_lattice)
        delta_E = new_E - old_E
        self._updateLattice(index, delta_E)
        self._updateTemperature()

    def run(self):     
        plt.ion()
        _, axs = plt.subplots(nrows=2, ncols=1)
        axs[1].set_title('Magnetization Curve')
        axs[1].set_xlabel('Temperature: {}'.format(self.temperature))  
        
        while ( self.isNotMagnetized() ):
            self._iterate()       
            self.render_plot(axs)
        
        input("Press Enter to exit...")

    def render_plot(self, axs):
        # Renders image only if it's in the last step count
        if self.iterations_per_temp_count == MAX_ITERATIONS_PER_TEMPERATURE:
            axs[0].clear()
            axs[0].set_title('Ising Model Simulation')
            axs[0].imshow(self.lattice,vmin=-1, vmax=1)
            
            axs[1].semilogx(self.temperature, self.getMagnetization(),"*b")
            axs[1].set_xlabel('Temperature: {}'.format(self.temperature))  
            plt.show()
            plt.pause(1e-12)


if __name__ == '__main__':
    isingModel = IsingModel(4,4,10)
    isingModel.run()
