import numpy as np

def generate_normalized_psd_matrix_case_1(d,n):
    """Generate a normalized PSD matrix with eigenvalues between 0 and 1."""
    A = np.random.rand(d, d) + 1j * np.random.rand(d, d)
    A = A + A.conj().T 
    eigenvalues, eigenvectors = np.linalg.eigh(A)
    normalized_eigenvalues = np.random.rand(2)/n
    return eigenvectors @ np.diag(normalized_eigenvalues) @ eigenvectors.conj().T

def generate_povm_set_case_1(d, n):
    """Generate a POVM set with specified properties."""
    povm_elements = [generate_normalized_psd_matrix_case_1(d,n) for _ in range(n)]
    
   
    total = sum(povm_elements)
  
    adjustment = np.eye(d) - sum(povm_elements) 
    povm_elements[-1] += adjustment
    
    return povm_elements

def generate_normalized_psd_matrix_case_2(d,n):
    """Generate a normalized PSD matrix with eigenvalues between 0 and 1."""
    A = np.random.rand(d, d) + 1j * np.random.rand(d, d)
    A = A + A.conj().T  
    eigenvalues, eigenvectors = np.linalg.eigh(A)
    normalized_eigenvalues = np.random.rand(2)
    return eigenvectors @ np.diag(normalized_eigenvalues) @ eigenvectors.conj().T

def generate_povm_set_case_2(d, n):
    """Generate a POVM set with specified properties."""
    povm_elements = [generate_normalized_psd_matrix_case_2(d,n) for _ in range(n)]
    adjustment = np.eye(d) - sum(povm_elements) 
    eigen_adj,_=np.linalg.eigh(adjustment)
    margin=0.9
    while((eigen_adj<0).any()):
        margin-=0.2
        
        total = sum(povm_elements)
        eigenval,eigenvec=np.linalg.eigh(total)
        
        correction_factor=max(eigenval)*(1-margin)
        povm_elements = [E / correction_factor for E in povm_elements]
        
   
        adjustment = np.eye(d) - sum(povm_elements) 
        eigen_adj,_=np.linalg.eigh(adjustment)
        
        
    povm_elements[-1] += adjustment
    
    return povm_elements

def is_a_proper_povm(set):
    for item in set:
        eigenval,_=np.linalg.eigh(item)
        if (eigenval<0).any():
            return False
    
    return True