import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

TEMPERATURE_DECREASING_FACTOR = 0.95
ITERATIONS_PER_TEMPERATURE = 1000

def sgn(n,ref=0):
		return 1 if n>=ref else -1

class IsingModel():

    def __init__(self, rows, cols, init_temp):
        self.rows = rows
        self.cols = cols
        self.lattice = np.vectorize(sgn)( np.random.rand(self.rows,self.cols), 0.5)
        self.temperature = init_temp
        self.iterations_per_temp_count = 0
        self.magnetization = self.getMagnetization()
        self.energy = self._getEnergy(self.lattice)

    def isNotMagnetized(self):
        self.magnetization = self.getMagnetization()
        if (abs(self.getMagnetization()) == self.cols*self.rows):
            return False
        else:
            return True

    def _accessLattice(self,indexes,lattice):
        row_index,col_index = indexes
        
        if (row_index >= self.rows) or (row_index < 0):
            return 0
        elif (col_index >= self.cols) or (col_index < 0):
            return 0
        else:
            return lattice[row_index,col_index]

    # Magnetization is simply the sum of all components in lattice
    def getMagnetization(self):
        return self.lattice.sum()

    def _updateTemperature(self):
        if self.iterations_per_temp_count == ITERATIONS_PER_TEMPERATURE:
            self.temperature *= TEMPERATURE_DECREASING_FACTOR
            self.iterations_per_temp_count = 1
        else:
            self.iterations_per_temp_count += 1


    def _getLatticeFlippedAt(self, index):
        new_lattice = np.copy(self.lattice)
        new_lattice[index] *= -1
        return new_lattice

    def _neighbourSum(self, index, lattice):
        i,j = index
        return (self._accessLattice((i+1,j),lattice)
                + self._accessLattice((i-1,j),lattice)
                + self._accessLattice((i,j+1),lattice) 
                + self._accessLattice((i,j-1),lattice)
                )

    def _getEnergy(self,lattice):
        energy = 0
        for i in range(self.rows):
            for j in range(self.cols):
                energy = energy-0.5 *lattice[i,j]*self._neighbourSum((i,j),lattice)
        return energy

    def _acceptanceProb(self, dE):
        try:
            probability = np.exp(-np.true_divide(dE,self.temperature))
            return probability
        except ZeroDivisionError:
            print("Catching ZeroDivisionError...")
            exit()

    def _acceptWithProb(self, delta_energy):
        if (np.random.random() <= self._acceptanceProb(delta_energy) ): # random() returns value between 0 and 1
            print("ACCEPT PROB: true")
            return True
        else:
            print("ACCEPT PROB: false")
            return False

    def _updateLattice(self, index, delta_energy):
        if delta_energy < 0:
            self.lattice[index] *= -1
            self.energy += delta_energy
            
            
        elif self._acceptWithProb(delta_energy): 
            self.lattice[index] *= -1
            self.energy += delta_energy
            
        # else: makes no change in lattice nor energy
            
       
    def _iterate(self):
        # Choose random element in lattice and get it flipped
        index = np.random.randint(self.rows), np.random.randint(self.cols)
        new_lattice = self._getLatticeFlippedAt(index)

        delta_E = self._getEnergy(new_lattice) - self.energy
        self._updateLattice(index, delta_E)
        self._updateTemperature()

    def run(self):     
        plt.ion()
        _, axs = plt.subplots(nrows=3, ncols=1)
        axs[1].set_title('Magnetization Curve')
        axs[1].set_xlabel('Temperature: {}. Energy: {}'.format(self.temperature,self.energy))  
        
        it = 0
        while ( self.isNotMagnetized() ):
            it += 1
            self._iterate()       
            self.render_plot(axs,it)
        self.render_plot(axs,it)
        input("Press Enter to exit...")

    def render_plot(self, axs,it):
        # Renders image only if it's in the last step count
        if self.iterations_per_temp_count == ITERATIONS_PER_TEMPERATURE:
            axs[0].clear()
            axs[0].set_title('Ising Model Simulation')
            axs[0].imshow(self.lattice,vmin=-1, vmax=1)
            
            axs[1].semilogx(self.temperature, self.magnetization,"*b")
            axs[2].plot(it,self.temperature,'*r') 
            axs[2].set_xlabel('Temperature: {}. Energy: {}'.format(self.temperature,self.energy)) 
            plt.show()
            plt.pause(1e-12)


if __name__ == '__main__':
    isingModel = IsingModel(10,10,5)
    isingModel.run()
