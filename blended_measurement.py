import numpy as np
from scipy.stats import unitary_group, ortho_group
from random_measurement import random_measurement 
from scipy.linalg import sqrtm

m=10
d=2
measurement_set=[]
for i in range(m):
    measurement_set.append(random_measurement(d,i))
# measurement_set

def blended_measurement(set,d,m):
    E=[]
    sum_set=np.zeros((d, d), dtype=np.complex128)
    for item in set:
        sum_set+=item
    E_0=sqrtm(1-sum_set/m)
    E.append(E_0)
    for item in set:
        E.append(sqrtm(item/m))
    return E

# blended_measurement(measurement_set,d)