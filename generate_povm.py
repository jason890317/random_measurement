import numpy as np
import random
from scipy.stats import unitary_group
from tools import generate_random_projector

def generate_normalized_psd_matrix(m,d,high=True):
    """Generate a normalized PSD matrix with eigenvalues between 0 and 1."""
    
    A = np.random.rand(d, d) + 1j * np.random.rand(d, d)
    A = A + A.conj().T 
    eigenvalues, eigenvectors = np.linalg.eigh(A)
    normalized_eigenvalues=[]
    if high:
        normalized_eigenvalues=[random.uniform(0.7, 1) for _ in range(d)]
    else:
        normalized_eigenvalues=[random.uniform(0.01, 0.2/m) for _ in range(d)]
        # print(len(normalized_eigenvalues))
    return eigenvectors @ np.diag(normalized_eigenvalues) @ eigenvectors.conj().T
    
        
def generate_povm_set_case_1(d, m):
    """Generate a POVM set with specified properties."""
    povm_elements = [generate_normalized_psd_matrix(m,d,False) for _ in range(m-1)]
    povm_elements.append(generate_normalized_psd_matrix(m,d,True))

    return povm_elements


def generate_povm_set_case_2(d, m):
    """Generate a POVM set with specified properties."""
    povm_elements = [generate_normalized_psd_matrix(m,d,False) for _ in range(m)]
    
    return povm_elements

def generate_povm_by_unitary_case_1(d,m,projector,roh,projector_random):
    
    if projector_random:
        projector=generate_random_projector(d)
    
    povm=[]
    
    while len(povm)!=m-1:
        U=unitary_group.rvs(d)
        temp=U@projector@U.T.conj()
        if 0.0001<np.trace(temp@roh)<0.2:
            povm.append(temp)
    while len(povm)!=m:
        U=unitary_group.rvs(d)
        temp=U@projector@U.T.conj()
        if 1>np.trace(temp@roh)>0.7:
            povm.append(temp)
        
    return povm

def generate_povm_by_unitary_case_2(d,m,projector,roh,projector_random):
    
    if projector_random:
        projector=generate_random_projector(d)

    povm=[]
    
    while len(povm)!=m:
        U=unitary_group.rvs(d)
        temp=U@projector@U.T.conj()
        if 0.0001<np.trace(temp@roh)<0.2/m:
            povm.append(temp)
    
    return povm