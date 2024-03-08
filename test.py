import numpy as np
def generate_random_statevector(d):
    # Generate a random complex vector
    vec = np.random.rand(d) + 1j * np.random.rand(d)
    # Normalize the vector
    statevector = vec / np.linalg.norm(vec)
    return statevector  


def generate_random_projector(d):
    # Generate a random complex vector
    vec = np.random.rand(d) + 1j * np.random.rand(d)
    
    # Normalize the vector
    vec_normalized = vec / np.linalg.norm(vec)
    
    # Construct the projector
    projector = np.outer(vec_normalized, np.conj(vec_normalized))
    
    return projector
p=generate_random_projector(5)
eig,vec=np.linalg.eig(p)
print(p)
print(eig)