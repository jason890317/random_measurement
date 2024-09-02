import numpy as np

def generate_first_vector(d,p):
    """
    Generate a vector of the form [a1, a2, ..., an, b1, b2, ..., bn] such that
    sum(|ai|^2) = A and sum(|bi|^2) = B without the constraint of positive real parts.
    
    Parameters:
    - n: Number of elements in a_i and b_i
    - A: Sum of squares of the magnitudes of a_i
    - B: Sum of squares of the magnitudes of b_i
    
    Returns:
    - vector: A numpy array containing the vector [a1, a2, ..., an, b1, b2, ..., bn]
    """
    
    n=int(d/2)
    
    # Generate random complex numbers with positive real parts for ai and bi
    real_a = np.random.rand(n)
    imag_a = np.random.randn(n)
    a = real_a + 1j * imag_a
    
    real_b = np.random.rand(n)
    imag_b = np.random.randn(n)
    b = real_b + 1j * imag_b
    
    # Normalize the vectors to satisfy the conditions
    norm_a = np.sqrt((1-p) / np.sum(np.abs(a)**2))
    norm_b = np.sqrt((p) / np.sum(np.abs(b)**2))
    
    a = norm_a * a
    b = norm_b * b
    
    # Ensure that all real parts of a and b are positive
    a = np.abs(a.real) + 1j * a.imag
    b = np.abs(b.real) + 1j * b.imag
    
    # Construct the final vector
    vector = np.concatenate([a, b])
    
    return vector

