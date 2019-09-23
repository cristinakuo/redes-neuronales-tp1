from hopfield import HopfieldNet
import numpy as np 
from tqdm import tqdm
from progress.spinner import Spinner
from utils import *

import logging
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s") 
log = logging.getLogger(__name__)

def main():
    P_error_ref = 0.001
    N = 100

    log.info("Start process...")
    step_size = 10

    results = list()

    pattern_count = 0
    P_error = 0
    with Spinner("Processing...") as bar:
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
            bar.next()
    log.info("Done")
    log.info("Error probability reached: {}/{} = {}".format(error_count,N*pattern_count, P_error))
    log.info("Learned pattern count - Neurons ratio: {}/{} = {}".format(
		pattern_count, N, np.true_divide(pattern_count,N)))


if __name__ == '__main__':
    main()