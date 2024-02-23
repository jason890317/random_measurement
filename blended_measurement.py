import numpy as np
from scipy.linalg import sqrtm

def blended_measurement(set,d,m):
    E=[]
    sum_set=np.zeros((d, d), dtype=np.complex128)
    for item in set:
        sum_set+=item
    E_0=sqrtm(1-sum_set/m)
    E_0=E_0.astype('complex128')
    E.append(E_0)
    for item in set:
        temp=sqrtm(item/m)
        temp=temp.astype('complex128')
        E.append(temp)
    return E
