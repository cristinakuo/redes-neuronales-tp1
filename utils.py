import numpy as np
from itertools import product


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

def sum_patterns(pattern_1, pattern_2):
    if len(pattern_1) != len(pattern_2):
        raise Exception("Sum of patterns: Length mismatch");
    else:
        result = pattern_1 + pattern_2
        result = np.vectorize(sgn)(result)
        return result

def invert_pattern(pattern):
    return np.array([x*(-1) for x in pattern])

def array_is_in(array, list_arrays):
    for arr in list_arrays:
        if np.array_equal(array,arr):
            return True
    return False

def get_all_combinations(patterns_list):
    combinations = []
    L = len(patterns_list)

    permutations_tuples = list(product([1,-1],repeat=L))
    sign_permutations = []
    for pr in permutations_tuples:
        sign_permutations.append(np.array(pr))

    if len(sign_permutations) != np.power(2,L):
         log.error("Permutations not doing right. Bye.")
         sys.exit(1)

    patterns_arr = np.array(patterns_list)
    for signs in sign_permutations:
        res = patterns_arr*signs[:,None]
        res = np.sum(res,axis=0)
        res = np.vectorize(sgn)(res)
        combinations.append(res)

    return combinations

def find_inverted_spurious(hopfieldNet,patterns_list):
    spurious_count = 0
    for pattern in patterns_list:
        refreshed = hopfieldNet.evaluate_net(invert_pattern(pattern),'async')
        if not array_is_in(refreshed,patterns_list):
            spurious_count += 1
    return spurious_count

def find_combinations_spurious(hopfieldNet,patterns_list):
    spurious_count = 0
    combinations = get_all_combinations(patterns_list)        
    
    for comb in combinations:
        refreshed = hopfieldNet.evaluate_net(comb,'async')
        if not array_is_in(refreshed,patterns_list):
            spurious_count += 1
    return spurious_count   