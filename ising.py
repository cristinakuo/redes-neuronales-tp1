import numpy as np
from utils import sgn

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.ion()

TEMPERATURE_DECREASING_FACTOR = 0.95
TEMPERATURE_MAX_STEP_COUNT = 2048

class IsingModel():

    def __init__(self, rows, cols, init_temp):
        self.rows = rows
        self.cols = cols
        # Inicializo lattice totalmente randomly
        self.lattice = np.vectorize(sgn)( np.random.rand(self.rows,self.cols), 0.5)
        self.temperature = init_temp
        self.temperature_step_count = 0
        self.magnetization = self.lattice.sum()

    def _bc_row(self, row_ind):
        if (row_ind >= self.rows) or (row_ind < 0):
            return 0
        else:
            return row_ind
    
    def _bc_col(self, col_ind):
        if (col_ind >= self.cols) or (col_ind < 0):
            return 0
        else:
            return col_ind

    def _update_magnetization(self):
        self.magnetization = self.lattice.sum()

    def _update_temperature(self):
        if self.temperature_step_count == TEMPERATURE_MAX_STEP_COUNT:
            self.temperature *= TEMPERATURE_DECREASING_FACTOR
            self.temperature_step_count = 0
            self.temperature_step_count += 1
        else:
            self.temperature_step_count += 1


    def _getLatticeWithFlippedCandidateAt(self, index):
        new_lattice = np.copy(self.lattice)
        new_lattice[index] *= -1
        return new_lattice

    def _get_cell_energy(self, index, lattice):
        i,j = index
        return -1/2*self.lattice[i,j]*(
                                self.lattice[self._bc_row(i+1),j] 
                                + self.lattice[self._bc_row(i-1),j] 
                                + self.lattice[i,self._bc_col(j+1)]
                                + self.lattice[i,self._bc_col(j-1)]
                                )

    def _get_energy(self,lattice):
        energy = 0
        for i in range(self.rows):
            for j in range(self.cols):
                energy = energy + self._get_cell_energy((i,j),lattice)
        return energy

    def _acceptance_prob(self, dE):
        p = 0
        try:
            p = np.exp(-dE/self.temperature)
        except ZeroDivisionError:
            log.warning("Catching ZeroDivisionError...")
        
        return p

    def _updateLattice(self, index, dE):
        if dE < 0:
            self.lattice[index] *= -1
        elif np.random.random() <= self._acceptance_prob(dE):
            self.lattice[index] *= -1
       
    def _iterate(self):
        index = np.random.randint(self.rows), np.random.randint(self.cols)
        new_lattice = self._getLatticeWithFlippedCandidateAt(index)

        old_E = self._get_energy(self.lattice)
        new_E = self._get_energy(new_lattice)

        self._updateLattice(index, new_E-old_E)
        self._update_magnetization()
        self._update_temperature()

    def run(self,axs):  
        while abs(self.magnetization != self.rows*self.cols):
            self._iterate()
            #self._render_plot(a)
            self.render_plot(axs, force=False)
        self.render_plot(axs)
        print(self.lattice)

    def render_plot(self, axs, force=True):
        # Se renderiza unicamente si se fuerza o esta en el ultimo paso
        if force or self.temperature_step_count == TEMPERATURE_MAX_STEP_COUNT:
            axs[0].clear()
            axs[0].set_title('Ising Model Simulation')
            axs[0].imshow(self.lattice,vmin=-1, vmax=1)
            axs[1].semilogx(self.temperature, self.magnetization, "ob")
            plt.show()
            plt.pause(1e-12)


if __name__ == '__main__':
    ismo = IsingModel(10,10,10)
    fig, axs = plt.subplots(1, 2)
    axs[1].set_title('Magnetization Curve')
    axs[1].set_xlabel('Temperature')
    
    while abs(ismo.magnetization != ismo.rows*ismo.cols):
        ismo._iterate()
        #self._render_plot(a)
        ismo.render_plot(axs, force=False)
    ismo.render_plot(axs)
    input("Press Enter to exit...")