import numpy as np


def sgn(n,ref=0):
		return 1 if n>=ref else -1

def add_noise(s,proportion):
    n = int(proportion*len(s))
    for i in np.random.permutation(len(s))[:n]:
        s[i] *= -1
    
    return s

def gen_random_pattern(N):
    U = np.random.uniform(0,1,N)
    arr = np.zeros(N)
    for i in range(N):
        if U[i] > 0.5:
            arr[i] = 1
        else:
            arr[i] = -1
    return arr

def gen_list_of_patterns(N_neurons, M_patterns):
    patterns_list = []
    for i in range(M_patterns):
        patterns_list.append(gen_random_pattern(N_neurons))
    return patterns_list