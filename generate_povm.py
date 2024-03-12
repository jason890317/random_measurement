import numpy as np
import random
from scipy.stats import unitary_group
import time
import sys
from tools import generate_rank_n_projector
def generate_normalized_psd_matrix(m,d,high=True):
    """Generate a normalized PSD matrix with eigenvalues between 0 and 1."""
    np.random.seed(int(time.time()))
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

def generate_povm_by_unitary_case_1(d,m,rank_h,rank_l,roh):
    
    povm=[]
    seed=int(time.time())
    len_povm=0
    
    projector_high=generate_rank_n_projector(rank_h,d)
    projector_low=generate_rank_n_projector(rank_l,d)
    
    while len(povm)!=m-1:
        np.random.seed(seed)
        U=unitary_group.rvs(d)
        temp=U@projector_low@U.T.conj()
        if 0.0000001<np.trace(temp@roh)<0.5:
            povm.append(temp)
        seed+=1
        if len_povm!=len(povm):
            sys.stdout.write(f"\rpovm : "+str(len_povm+1)+"/"+str(m))
            sys.stdout.flush()
            
        len_povm=len(povm)
    while len(povm)!=m:
        np.random.seed(seed)
        U=unitary_group.rvs(d)
        temp=U@projector_high@U.T.conj()
        if 1>np.trace(temp@roh)>0.7:
            povm.append(temp)
        seed+=1
        if len_povm!=len(povm):
            sys.stdout.write(f"\rpovm : "+str(len_povm+1)+"/"+str(m))
            sys.stdout.flush()
            
        len_povm=len(povm)
    print()
    return povm

def generate_povm_by_unitary_case_2(d,m,rank,roh):
   
    povm=[]
    seed=int(time.time())
    len_povm=0
    
    projector=generate_rank_n_projector(rank,d)
    
    while len(povm)!=m:
        np.random.seed(seed)
        U=unitary_group.rvs(d)
        temp=U@projector@U.T.conj()
        if 0.0000001<np.trace(temp@roh)<0.3/m:
            povm.append(temp)
        seed+=1
        if len_povm!=len(povm):
            sys.stdout.write(f"\rpovm : "+str(len_povm+1)+"/"+str(m))
            sys.stdout.flush()
            
        len_povm=len(povm)
    
    print()
    return povm