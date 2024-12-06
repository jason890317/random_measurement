import numpy as np
from scipy.stats import unitary_group
from scipy.integrate import quad
from scipy.stats import rv_continuous
import sys
from generate_vector import generate_first_vector

def generate_povm_general(case, d, m, rank, case_1_h, case_1_l, case_2_l, rho):
    """
    Generate a POVM (Positive Operator-Valued Measure) by rotating projectors.

    Parameters:
    - case (int): Case type (1, 2, or 3).
    - d (int): Dimension of the Hilbert space.
    - m (int): Number of POVM elements.
    - rank (int): Rank of the projectors.
    - case_1_h (float): High probability for case 1.
    - case_1_l (float): Low probability for case 1.
    - case_2_l (float): Low probability for case 2.
    - rho (np.ndarray): Density matrix.
  
    Returns:
    - list: List of POVM elements.
    """
    # Define the number of high-probability and low-probability elements
    if case == 1:  # Case 1: Single high-probability POVM
        number_of_high = 1
        high_pro = case_1_h
        low_pro = case_1_l
    elif case == 3:  # Case 3: Half high-probability, half low-probability POVM
        number_of_high = int(m / 2)
        high_pro = case_1_h
        low_pro = case_1_l
    elif case == 2:  # Case 2: Only low-probability POVM
        number_of_high = 0
        high_pro = 1
        low_pro = case_2_l

    number_of_low = m - number_of_high  # Calculate the remaining low-probability elements
    povm = []  # List to store POVM elements

    # Define the base projector matrix
    projector = np.diag(np.hstack([np.zeros(d - rank), np.ones(rank)]))

    def generate_povm_elements(number, prob_range, condition):
        """
        Add POVM elements to the list by generating random unitary matrices.

        Parameters:
        - number (int): Number of elements to generate.
        - prob_range (tuple): Range of probabilities for generation.
        - condition (function): Condition for accepting generated elements.
        """
        while len(povm) < number:
            p = distribution(*prob_range)  # Sample a probability value
            epson_vector = generate_first_vector(d, p, rank)  # Generate a random vector
            U = yield_random_unitary(d, epson_vector)  # Generate a random unitary matrix with special first column
            temp = U.T.conj() @ projector @ U  # Rotate the projector

            # Check if the generated matrix is a valid POVM element
            if np.allclose(temp @ temp, temp, atol=1e-14) and condition(np.trace(temp @ rho)):
                povm.append(temp)  # Add to the POVM set if valid

            # Update the progress in the terminal
            sys.stdout.write(f"\rPOVM: {len(povm)}/{m} with trace(temp @ rho): {np.trace(temp @ rho)}")
            sys.stdout.flush()

    # Generate low-probability POVM elements
    generate_povm_elements(number_of_low, (0.003, low_pro), lambda trace: trace < low_pro)

    # Generate high-probability POVM elements
    generate_povm_elements(m, (high_pro, 1), lambda trace: trace > high_pro)

    return povm



def yield_random_unitary(d, epson_vector):
    """
    Generate a random unitary matrix with a specified first column. 
    The explicit explanation is demonstrated in "Simulation Result of Random Measurements"
    
    Parameters:
    d (int): Dimension of the unitary matrix.
    epson_vector (np.ndarray): Vector to set as the first column.

    Returns:
    np.ndarray: Random unitary matrix.
    """
    A = unitary_group.rvs(d)
    A[:, 0] = epson_vector
    U, _ = np.linalg.qr(A)
    return U

def distribution(a, b):
    """
    Generate a sample from a custom distribution over the interval [a, b].

    Parameters:
    a (float): Lower bound of the interval.
    b (float): Upper bound of the interval.

    Returns:
    float: Sample from the custom distribution.
    """
    def normal_pdf(x, mu=0.5, sigma=0.1):
        return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

    normalization_constant = quad(normal_pdf, a, b)[0]

    def normalized_custom_pdf(x):
        return normal_pdf(x) / normalization_constant

    class CustomDistribution(rv_continuous):
        def _pdf(self, x):
            return normalized_custom_pdf(x)

    custom_dist = CustomDistribution(a=a, b=b, name='custom')
    sample = custom_dist.rvs(size=1)
    return sample
