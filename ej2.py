from hopfield import HopfieldNet
import numpy as np 
from tqdm import tqdm
from progress.spinner import Spinner
from utils import *
from tabulate import tabulate

import logging
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s") 
log = logging.getLogger(__name__)

def get_capacity_with_Perror_ref(P_error_ref):
    

    N=100

    log.info("Start process...")
    step_size = 10

    results = list()

    pattern_count = 0
    P_error = 0
    #with Spinner("Processing...") as bar:
    while P_error < P_error_ref:
        pattern_count += 1
        patterns = gen_list_of_patterns(N,pattern_count)
        myHop = HopfieldNet()
        for p in patterns:
            myHop.load_pattern_arr(p)
        myHop.train()
        errors = []
        for p in patterns:
            errors.append(myHop.refresh_synchronic(p, render=False) - p)
            
        errors = np.vstack(errors)
        error_count = len(np.flatnonzero(errors))
        P_error = np.true_divide(error_count, N * pattern_count)
            #bar.next()
    capacity = np.true_divide(pattern_count,N)
    

    return (P_error, capacity)

def main():
    P_errors_ref = [0.001, 0.0036, 0.01, 0.05, 0.1]

    results = []
    for p in P_errors_ref:
        results.append(get_capacity_with_Perror_ref(p))

    print(tabulate(results, headers=['P_err reached', 'Capacity']))

if __name__ == '__main__':
    main()