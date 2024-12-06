import numpy as np

def generate_first_vector(d, p, rank):
    """
    Generate a complex vector that satisfies specific normalization conditions.
    
    Parameters:
    - d (int): Total dimension of the Hilbert space.
    - p (float): Fraction of the vector's magnitude allocated to the 'b' components.
    - rank (int): Number of elements in the 'b' component.
    
    Returns:
    - vector (np.ndarray): A complex vector satisfying the given constraints.
    """
    # Divide dimensions into two parts: 'a' and 'b'
    n = d - rank  # Number of elements in 'a'
    m = rank      # Number of elements in 'b'

    # Generate random complex numbers for 'a' and 'b'
    real_a, imag_a = np.random.rand(n), np.random.randn(n)  # Real and imaginary parts for 'a'
    a = real_a + 1j * imag_a  # Complex vector 'a'

    real_b, imag_b = np.random.rand(m), np.random.randn(m)  # Real and imaginary parts for 'b'
    b = real_b + 1j * imag_b  # Complex vector 'b'

    # Normalize 'a' and 'b' to satisfy the magnitude constraints
    norm_a = np.sqrt((1 - p) / np.sum(np.abs(a)**2))  # Scale factor for 'a'
    norm_b = np.sqrt(p / np.sum(np.abs(b)**2))       # Scale factor for 'b'
    a = norm_a * a  # Normalize 'a'
    b = norm_b * b  # Normalize 'b'

    # Ensure real parts of all elements are positive
    a = np.abs(a.real) + 1j * a.imag
    b = np.abs(b.real) + 1j * b.imag

    # Combine 'a' and 'b' into a single vector
    vector = np.concatenate([a, b])

    return vector
