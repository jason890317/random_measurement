import numpy as np
import random
def generate_normalized_psd_matrix_case_1(d,high=True):
    """Generate a normalized PSD matrix with eigenvalues between 0 and 1."""
    
    A = np.random.rand(d, d) + 1j * np.random.rand(d, d)
    A = A + A.conj().T 
    eigenvalues, eigenvectors = np.linalg.eigh(A)
    normalized_eigenvalues=[]
    if high:
        normalized_eigenvalues=[random.uniform(0.7, 1) for _ in range(2)]
    else:
        normalized_eigenvalues=[random.uniform(0.00001, 0.2) for _ in range(2)]
    return eigenvectors @ np.diag(normalized_eigenvalues) @ eigenvectors.conj().T
    
        
def generate_povm_set_case_1(d, m):
    """Generate a POVM set with specified properties."""
    povm_elements = [generate_normalized_psd_matrix_case_1(d,False) for _ in range(m-1)]
    povm_elements.append(generate_normalized_psd_matrix_case_1(d,True))
   
    # total = sum(povm_elements)
  
    # adjustment = np.eye(d) - sum(povm_elements) 
    # povm_elements[-1] += adjustment
    
    return povm_elements


def generate_povm_set_case_2(d, m):
    """Generate a POVM set with specified properties."""
    povm_elements = [generate_normalized_psd_matrix_case_1(d,False) for _ in range(m)]
    
    return povm_elements

