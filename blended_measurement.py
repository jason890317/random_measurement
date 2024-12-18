import numpy as np
from scipy.linalg import sqrtm

def _compute_optimizing_factor(set, d, m):
    """Compute the optimizing factor to balance matrix scaling."""
    sum_set = np.sum(set, axis=0)
    eigenvalues = np.linalg.eigvals(sum_set)
    return 1 / max(np.abs(eigenvalues) / m)

def blended_measurement(set, d, m):
    """
    Generate measurement operators using blended measurement procedure.
    
    Args:
        set (list): List of input matrices
        d (int): Matrix dimension
        m (int): Scaling parameter
    
    Returns:
        list: Measurement operators
    """
    sum_set = np.sum(set, axis=0)
    identity = np.eye(d)
    
    # Compute first operator (the outcome zero)
    E_0 = sqrtm(identity - sum_set/m).astype(np.complex128)
    
    # Compute other outcome operators
    E = [E_0] + [sqrtm(item/m).astype(np.complex128) for item in set]
    
    # Make operators into Measureements
    blended_set=[ item@item.T.conj() for item in E]
    
    return blended_set

def inverse_blended_measurement(set, d, m):
    """
    Generate measurement operators using inverse blended measurement.
    
    Args:
        set (list): List of input matrices
        d (int): Matrix dimension
        m (int): Scaling parameter
    
    Returns:
        list: Measurement operators
    """
    identity = np.eye(d)
    sum_set = np.sum(identity - set, axis=0)
    
    # Compute first operator (zero outcome)
    E_0 = sqrtm(identity - sum_set/m).astype(np.complex128)
    
    # Compute other outcome operators
    E = [E_0] + [sqrtm((identity - item)/m).astype(np.complex128) for item in set]
    
    # Make operators into Measureements
    inverse_blended_set=[ item@item.T.conj() for item in E]
    
    
    return inverse_blended_set

def optimizing_blended_measurement(set, d, m):
    """
    Generate optimized measurement operators by scaling matrices.
    
    Args:
        set (list): List of input matrices
        d (int): Matrix dimension
        m (int): Scaling parameter
    
    Returns:
        list: Optimized measurement operators
    """
    # Compute optimizing factor to balance matrix scaling
    optimizing_factor = _compute_optimizing_factor(set, d, m)
    
    identity = np.eye(d)
    sum_set = np.sum(set, axis=0)
    
    # Compute first operator (the outcome zero)
    E_0 = sqrtm(identity - (sum_set/m) * optimizing_factor).astype(np.complex128)
    
    # Compute other outcome operators
    E = [E_0] + [sqrtm((item/m) * optimizing_factor).astype(np.complex128) for item in set]
    
    # Make operators into Measureements
    optimizing_blended_set=[ item@item.T.conj() for item in E]
    
    return optimizing_blended_set

def three_outcome_blended_measurement(povm_set, d, m,permutations):
    """
    Generate measurement operators for the three-outcome blended measurement procedure.

    Args:
        povm_set (list): List of POVM elements.
        d (int): Matrix dimension.
        m (int): Number of POVM elements in the set.

    Returns:
        list: Three-outcome blended measurement sets.
    """
   
    # Precompute the first operator (E_0)
    sum_set = np.sum(povm_set, axis=0)
    identity = np.eye(d)
    E_0 = sqrtm(identity - sum_set/m).astype(np.complex128)

    # Generate the blended measurement sets
    three_outcome_blended_set = []
    for round_itr in range(int(5 * m)):
        sum_set_1 = np.zeros((d, d), dtype=np.complex128)
        sum_set_2 = np.zeros((d, d), dtype=np.complex128)
        E = [E_0]  # Start with E_0
        
        # Every three-outcome blended measurement set is generated by a permutation
        for i in range(m):
            if permutations[round_itr][i] == 1:
                sum_set_1 += povm_set[i]
            elif permutations[round_itr][i] == 2:
                sum_set_2 += povm_set[i]

        # Compute operators of three-outcome bledned measurement 
        temp_1 = sqrtm(sum_set_1/m).astype(np.complex128)
        temp_2 = sqrtm(sum_set_2/m).astype(np.complex128)
        E.append(temp_1)
        E.append(temp_2)

        # Store each three-outcome set (one three-outcome measurement set created by different permutation) 
        one_three_outcome_set = [item @ item.T.conj() for item in E]
        three_outcome_blended_set.append(one_three_outcome_set)

    return three_outcome_blended_set