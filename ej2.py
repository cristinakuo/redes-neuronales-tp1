from hopfield import *
import numpy as np 
from tqdm import tqdm
from progress.spinner import Spinner
from utils import *
from tabulate import tabulate

import logging
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s") 
log = logging.getLogger(__name__)

# Check Hopfield Net's capacity statistically, for different admisible errors
def main():
    P_errors_ref = [0.001, 0.0036, 0.01, 0.05, 0.1]

    capacities = []
    for p in P_errors_ref:
        capacity = get_capacity(neurons=100, P_error_ref=p)
        capacities.append(capacity)

    results_table = tabulate(capacities, headers=['P_err reached', 'Capacity'])
    print(results_table)

if __name__ == '__main__':
    main()