import numpy as np
from scipy.stats import unitary_group
from scipy.integrate import quad
from scipy.stats import rv_continuous
from vector_generating_function import generate_first_vector

def generate_povm_general(case, dimension_of_povm, num_of_measurements, rank, case_1_high_probability, case_1_low_probability, case_2_low_probability, rho):
    """
    Generate a POVM (Positive Operator-Valued Measure) by rotating projectors.

    Parameters:
    - case : Case type ("1", "2", or "special").
    - dimension_of_povm (int): Dimension of the Hilbert space.
    - num_of_measurements (int): Number of POVM elements.
    - rank (int): Rank of the projectors.
    - case_1_high_probability (float): High probability for case 1.
    - case_1_low_probability (float): Low probability for case 1.
    - case_2_low_probability (float): Low probability for case 2.
    - rho (np.ndarray): Density matrix.
  
    Returns:
    - list: List of POVM elements.
    """
    
    # Define the number of high-probability and low-probability elements
    if case == "1":  # Case 1: Single high-probability POVM
        num_of_high_probability_measurment = 1
        threshold_high_probability = case_1_high_probability
        threshold_low_probability = case_1_low_probability
    elif case == "2":  # Case 2: Only low-probability POVM
        num_of_high_probability_measurment = 0
        threshold_high_probability = 1
        threshold_low_probability = case_2_low_probability
    elif case == "special":  # Case special: Half high-probability, half low-probability POVM
        num_of_high_probability_measurment = int(num_of_measurements / 2)
        threshold_high_probability = case_1_high_probability
        threshold_low_probability = case_1_low_probability
    
    # Calculate the remaining low-probability elements
    num_of_low_probability_measurement = num_of_measurements - num_of_high_probability_measurment  
   

    # Initialize the povm set
    povm_set=[]
   
    # Generate low-probability POVM elements
    generate_povm_elements(povm_set,num_of_low_probability_measurement, dimension_of_povm,rank,rho, prob_range=(0.003, threshold_low_probability), condition=lambda trace: trace < threshold_low_probability)

    # Generate high-probability POVM elements
    generate_povm_elements(povm_set,num_of_high_probability_measurment,dimension_of_povm,rank,rho, prob_range=(threshold_high_probability, 1), condition=lambda trace: trace > threshold_high_probability)
    
   
    return povm_set



def generate_povm_elements(povm_set,number, dimension_of_povm, rank, rho, prob_range, condition):
        """
        Add POVM elements to the list by generating random unitary matrices.

        Parameters:
        - number (int): Number of elements to generate.
        - prob_range (tuple): Range of probabilities for generation.
        - condition (function): Condition for accepting generated elements.
        """        
        projector = np.diag(np.hstack([np.zeros(dimension_of_povm - rank), np.ones(rank)]))

        for _ in range(number):
            p = distribution(*prob_range)  # Sample a probability value
            epson_vector = generate_first_vector(dimension_of_povm, p, rank)  # Generate a random vector
            U = yield_random_unitary(dimension_of_povm, epson_vector)  # Generate a random unitary matrix with special first column
            temp = U.T.conj() @ projector @ U  # Rotate the projector

            # Check if the generated matrix is a valid POVM element
            if np.allclose(temp @ temp, temp, atol=1e-14) and condition(np.trace(temp @ rho)):
                povm_set.append(temp)  # Add to the POVM set if valid

            
        
def yield_random_unitary(dimension_of_povm, epson_vector):
    """
    Generate a random unitary matrix with a specified first column. 
    The explicit explanation is demonstrated in "Simulation Result of Random Measurements"
    
    Parameters:
    dimension_of_povm (int): Dimension of the unitary matrix.
    epson_vector (np.ndarray): Vector to set as the first column.

    Returns:
    np.ndarray: Random unitary matrix.
    """
    A = unitary_group.rvs(dimension_of_povm)
    A[:, 0] = epson_vector
    U, _ = np.linalg.qr(A)
    return U

def distribution(a, b):
    """
    Generate a sample from a custom distribution over the interval [a, b]. Truncated normal distribution is used.

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
