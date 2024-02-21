import numpy as np
# from scipy.stats import unitary_group, ortho_group

def random_measurement(d,seed):
    np.random.seed(seed)
    # Generate a random complex matrix
    A = np.random.rand(d, d) +1j * np.random.rand(d, d)
    
    # Make the matrix Hermitian by adding it to its conjugate transpose and then normalizing
    H = (A + A.conj().T) / 2
    
    # Ensure the eigenvalues are between 0 and 1 by normalizing the matrix
    eigval, eigvec = np.linalg.eig(H)
    eigval=np.random.rand(d)  # 2 random eigenvalues between 0 and 1
    eigval = np.diag(eigval)
    H_normalized= eigvec@eigval@eigvec.T.conj()
    return H_normalized
    
# for i in range(10):
#     a=random_measurement(3,i)
#     # print("matrix : ", a)
#     eigvl,eigvec=np.linalg.eig(a)     
#     print("eigenvalue : ",eigvl)
    