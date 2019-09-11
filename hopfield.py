import logging
import time 
import image
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from others import sgn
from others import add_noise

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s") 
log = logging.getLogger(__name__) # TODO: whats this

class HopfieldNet:
    def __init__(self):
        self.rows = None
        self.cols = None
        self.patterns = []
        self.W = np.zeros(0) # Synaptic weight matrix
        self.N = 0 # Number of neurons
        self.max_iteration = 20

    def load_pattern(self, file_name):
        log.info("Loading binary file: '{}'...".format(file_name))
        img = image.load_binary_image(file_name)
        cols, rows = img["cols"], img["rows"]

        if not self.cols and not self.rows:
            log.info("Setting dimensions: '{}x{}' (ROWSxCOLS).".format(rows,cols))
            self.cols, self.rows = cols, rows
            self.N = cols * rows
            log.info("Using N={} neurons.".format(self.N))
        else:
            if self.cols != cols or self.rows != rows:
                raise Exception("Dimensions mismatch: '{}x{}' - '{}x{} (ROWSxCOLS)'".format(self.cols, self.rows, rows, cols))

        self.patterns.append(img["data"])

    def load_patterns(self, file_names):
        log.info("Loading training images...")
        for f in file_names:
            self.load_pattern(f)

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

         
    def refresh_net(self, s, render):   # Asynchronic     
        if render:
            im, bitmap = image.render_image(s, self.rows, self.cols)
        
        refreshed_s = np.copy(s)
        
        for i in tqdm(np.random.permutation(self.N)):
            refreshed_s[i] = sgn(sum(self.W[i,:]*refreshed_s))
            
            if render and refreshed_s[i] != s[i]:
                im, bitmap = image.render_pixel(im, bitmap, refreshed_s[i], i%self.cols, int(np.floor(i/self.cols)))
        return refreshed_s

    def get_energy(self,s):
        if not self.W.any():
            raise("No synaptic wighttrained.")
        
        if len(s) != self.N:
            raise("Dimensions mismatch")

        return -0.5*sum(np.dot(self.W, s))

    def evaluate_net(self, s, render=False):
        if not self.W.any():
            raise("No synaptic weight matrix trained.")
        
        if len(s) != self.N:
            raise("Dimensions mismatch.")

        current_H = 0
        previous_H = 0
        for i in range(self.max_iteration):
            log.info("Iteration {}".format(i+1))
            
            log.info("Refreshing net...")    
            s = self.refresh_net(s, render=render)
            
            log.info("Calculating energy...")
            previous_H = current_H
            current_H = self.get_energy(s)
            
            if current_H == previous_H:
                log.info("Reached minimum energy ('{}').".format(current_H))
                break
        else:
            log.warning("Reached maximum iteration count ({}).".format(self.max_iteration))
        return s
        
    def test(self,testing_set):
        log.info("Testing patterns...")
        for test in testing_set:

            s = image.load_binary_image(test)["data"]
            
            s = add_noise(s, 0.25)
            image.render_image(s,self.rows,self.cols)
            input("Press enter to continue...")
            
            net = self.evaluate_net(s,render=True)
            image.render_image(net, self.rows, self.cols)
            input("Press enter to continue...")
            



def main():
    plt.ion()

    training_set = [
        "img/panda.bmp",
        "img/v.bmp",
        "img/perro.bmp"
    ]
    testing_set = [
        "img/panda.bmp"
    ]

    myHop = HopfieldNet()
    myHop.load_patterns(training_set)
    myHop.train()
    myHop.test(testing_set)

if __name__ == '__main__':
    main()
