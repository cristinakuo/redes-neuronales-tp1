import logging
import time 
import image
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils import *

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s") 
log = logging.getLogger(__name__) 

class HopfieldNet:
    def __init__(self):
        self.rows = None
        self.cols = None
        self.patterns = []
        self.W = np.zeros(0) # Synaptic weight matrix
        self.N = None # Number of neurons

    def load_image_pattern(self, file_name):
        log.info("Loading binary file: '{}'...".format(file_name))
        img = image.load_binary_image(file_name)
        cols, rows = img["cols"], img["rows"]

        if not self.cols and not self.rows:
            log.info("Setting dimensions: '{}x{}' (ROWSxCOLS).".format(rows,cols))
            self.cols, self.rows = cols, rows
            self.N = cols * rows
            log.info("Using N={} neurons.".format(self.N))
        elif self.cols != cols or self.rows != rows:
                raise Exception("Dimensions mismatch: '{}x{}' - '{}x{} (ROWSxCOLS)'".format(self.cols, self.rows, rows, cols))

        self.load_pattern_arr(img["data"])

    def load_image_patterns(self, file_names):
        log.info("Loading training images...")
        for f in file_names:
            self.load_image_pattern(f)

    # Sin imagen
    def load_pattern_arr(self, p):
        if not self.N:
            self.N = len(p)
            log.info("Using N={} neurons.".format(self.N))
        elif len(p) != self.N:
                raise Exception("Length of patterns mismatch: '{}' - '{}'.".format(self.N,len(p)))
        
        self.patterns.append(p)

    def load_patterns_list(self,patterns_list):
        for p in patterns_list:
            self.load_pattern_arr(p)

    def train(self):
        log.info("Training net...")
        def _get_Wij(i,j):
            if i == j:
                return 0.0
            
            if i>self.N or j > self.N:
                raise Exception("Index out of range.")

            return sum(p[i]*p[j] for p in self.patterns)/len(self.patterns)
        if not self.patterns:
            raise Exception("No patterns provided.")
        
        self.W = np.zeros((self.N,self.N))
        for i in tqdm(range(self.N)):
            for j in range(self.N):
                self.W[i,j] = _get_Wij(i,j)

         
    def refresh_asynchronic(self, s, render):   # Asynchronic     
        if render:
            im, bitmap = image.render_image(s, self.rows, self.cols)
        
        refreshed_s = np.copy(s)
        
        for i in (np.random.permutation(self.N)):
            refreshed_s[i] = sgn(sum(self.W[i,:]*refreshed_s))
            
            if render and refreshed_s[i] != s[i]:
                im, bitmap = image.render_pixel(im, bitmap, refreshed_s[i], i%self.cols, int(np.floor(i/self.cols)))
        return np.array(refreshed_s)

    def refresh_synchronic(self, s, render):
        refreshed = [sgn(x) for x in np.dot(self.W,s)]
        # TODO: ver eso de render image
        if render:
            im, bitmap = image.render_image(refreshed, self.rows, self.cols)
        
        return np.array(refreshed)

    def get_energy(self,s):
        if not self.W.any():
            raise("No synaptic wighttrained.")
        
        if len(s) != self.N:
            raise("Dimensions mismatch")

        return -0.5*sum(np.dot(self.W, s))

    def evaluate_net(self, s, refresh_type='async', max_iteration=20, render=False):
        if not self.W.any():
            raise("No synaptic weight matrix trained.")
        
        if len(s) != self.N:
            raise("Dimensions mismatch.")

        if refresh_type == 'async':
            refreshed = self.refresh_asynchronic(s,render)
        elif refresh_type == 'sync':
            refreshed = self.refresh_synchronic(s,render)
        else:
            raise("Not a refresh type.")
        
        return refreshed

    
        current_H = 0
        previous_H = 0
        for i in range(max_iteration):
            #log.info("Iteration {}".format(i+1))
            
            #log.info("Refreshing net...")    
            s = refresh(s, render=render)
            
            #log.info("Calculating energy...")
            previous_H = current_H
            current_H = self.get_energy(s)
            
            if current_H == previous_H:
                log.info("Reached minimum energy ('{}').".format(current_H))
                break
        else:
            log.warning("Reached maximum iteration count ({}).".format(self.max_iteration))
        return s
    
    def get_error(self, s, refreshed):
        n_wrong_bits = np.count_nonzero(s-refreshed)
        return n_wrong_bits/self.N

    # TODO: ver donde lo uso
    def test_patterns(self,testing_patterns):
        log.info("Testing patterns...")
        for s in testing_patterns:
            s_noisy = add_noise(s, 0.25)
            s_refreshed = self.evaluate_net(s_noisy,render=False)
            log.info("Error rate is: '{}'".format(self.get_error(s,s_refreshed)))

    def random_disconnect(self, proportion):
        row_index = np.random.permutation(range(self.N))
        col_index = np.random.permutation(range(self.N))
        lim = int(np.floor(proportion*self.N))
        for (i,j) in zip(row_index[:lim],col_index[:lim]):
            self.W[i,j] = 0


def get_capacity(neurons,P_error_ref, disconnect_proportion=0): 
    N=neurons # Number of neurons
    step_size = 10

    results = list()

    pattern_count = 0
    P_error = 0
    while P_error < P_error_ref:
        pattern_count += 1
        patterns = gen_list_of_patterns(N,pattern_count)
        myHop = HopfieldNet()
        for p in patterns:
            myHop.load_pattern_arr(p)
        
        myHop.train()
        myHop.random_disconnect(disconnect_proportion)

        errors = []
        for p in patterns:
            errors.append(myHop.refresh_synchronic(p, render=False) - p)
            
        errors = np.vstack(errors)
        error_count = len(np.flatnonzero(errors))
        P_error = np.true_divide(error_count, N * pattern_count)
            #bar.next()
    capacity = np.true_divide(pattern_count,N)
    

    return (P_error, capacity)    
        
if __name__ == '__main__':
    ej_1()
