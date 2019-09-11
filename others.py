import numpy as np

def sgn(n):
		return 1 if n>=0 else -1

def add_noise(s,proportion):
    n = int(proportion*len(s))
    for i in np.random.permutation(len(s))[:n]:
        s[i] *= -1
    
    return s