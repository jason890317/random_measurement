import numpy as np
from tools import (
    compute_probability_for_povm_set, check_outcome_condition_for_blended_case_1, check_outcome_condition_for_blended_case_2,
    check_outcome_condition_for_random_case_special, check_outcome_condition_for_random_case_1, check_outcome_condition_for_blended_case_special,
    check_outcome_condition_for_interweave_case_special, generate_permutations, check_outcome_condition_for_blended_three_case_special,
    check_outcome_condition_for_random_case_2
)
from blended_measurement import (
    blended_measurement, inverse_blended_measurement, optimizing_blended_measurement, three_outcome_blended_measurement
)
from circuit import (
    blended_circuit, random_sequences_circuit, run_circuit,
    interweave_blended_circuit, three_outcome_blended_circuit
)

def quantum_event_identification(copies, d, m, gate_num_times, povm_set, rho, case, state, test_time, case_1_high, method):
    """
    Solving quantum event identification with five methods: "special random", "special blended", "optimizing blended", "interweave", "blended_three".

    Parameters:
    - copies: Number of repetitions.
    - d: Dimension of the system.
    - m: Number of POVM elements.
    - gate_num_times: Scaling factor for gates in blended methods.
    - povm_set: List of POVM elements.
    - rho: Density matrix of the system.
    - case: Test case "Special".
    - state: Initial quantum state.
    - test_time: Number of test iterations.
    - case_1_high: High probability threshold.
    - method: Method to use ('special_random', 'special_blended', etc.).

    Returns:
    - result: Dictionary containing experimental results.
    """
    accept_time = 0  # Count of successful outcomes

    if method == 'special_random':
        # Simulate using the 'special_random' method
        counts_set, shuffled_indices_set = start_simulation(
            random_sequences_circuit, copies, povm_set, state, m, case_1_high, include_indices=True
        )
        accept_time = check_outcome_condition_for_random_case_special(counts_set, shuffled_indices_set, m)

    elif method in ['special_blended', 'optimizing_blended']:
        # Simulate using the 'special_blended' or 'optimizing_blended' method
        blended_set = (optimizing_blended_measurement(povm_set, d, m) if method == "optimizing_blended" 
                       else blended_measurement(povm_set, d, m))
        counts_set = start_simulation(blended_circuit, copies, blended_set, state, int(gate_num_times * m))
        accept_time = check_outcome_condition_for_blended_case_special(counts_set, m, gate_num_times)

    elif method == "interweave":
        # Simulate using the 'interweave' method
        blended_set = blended_measurement(povm_set, d, m)
        blended_set_inv = inverse_blended_measurement(povm_set, d, m)
        counts_set = start_simulation(interweave_blended_circuit, copies, blended_set, blended_set_inv, state, int(gate_num_times * m))
        accept_time = check_outcome_condition_for_interweave_case_special(counts_set, m, gate_num_times)

    elif method == "blended_three":
        # Simulate using the 'blended_three' method
        permutation = generate_permutations(m, int(5 * m))
        three_outcome_blended_set = three_outcome_blended_measurement(povm_set, d, m, permutation)
        counts_set = start_simulation(three_outcome_blended_circuit, copies, three_outcome_blended_set, state, int(5 * m))
        accept_time = check_outcome_condition_for_blended_three_case_special(counts_set, m, permutation, int(5 * m))

    experiment = accept_time / m  # Calculate experimental probability
    return {"theorem": 0, "experiment": experiment}


def quantum_event_finding(copies, d, m, gate_num_times, povm_set, rho, case, state, test_time, case_1_high, method):
    """
    Implementing quantum event finding with methods: "random" and "blended", proposed by "quantum event learning and gentle random measurements".

    Parameters:
    - Case: 1 or 2

    Returns:
    - result: Dictionary containing theoretical and experimental results.
    """
    accept_time = 0  # Count of successful outcomes
    fail_time = 0  # Count of failed outcomes
    povm_set_probability = compute_probability_for_povm_set(povm_set, rho, False)
    povm_set_probability = sorted(povm_set_probability, reverse=True)

    if method == 'random':
        
        # Compute theoretical bounds for 'random' method
        epsilon = 1 - povm_set_probability[0]
        beta = sum(povm_set_probability[1:])
        at_least_pro = ((1 - epsilon) ** 7) / (1296 * ((1 + beta) ** 7))
        delta = 2 * sum(povm_set_probability)

        # Simulate the 'random' method
        for _ in range(test_time):
            
            counts_set, indices = start_simulation(
                random_sequences_circuit, copies, povm_set, state, m, case_1_high, include_indices=True
            )
            
            # check the acceptance of the result
            if case == "1" and check_outcome_condition_for_random_case_1(counts_set, indices):
                accept_time += 1
            elif case == "2" and check_outcome_condition_for_random_case_2(counts_set):
                fail_time += 1

    elif method == 'blended':
        
        # Compute theoretical bounds for 'blended' method
        epsilon = 1 - povm_set_probability[0]
        beta = sum(povm_set_probability[1:])
        at_least_pro = ((1 - epsilon) ** 3) / (12 * (1 + beta))
        delta = sum(povm_set_probability)

        # Simulate the 'blended' method
        blended_set = blended_measurement(povm_set, d, m)
        for _ in range(test_time):
            counts_set = start_simulation(blended_circuit, 1, blended_set, state, m)
            
            # check the acceptance of the result 
            if case == "1" and check_outcome_condition_for_blended_case_1(counts_set, m):
                accept_time += 1
            elif case == "2" and check_outcome_condition_for_blended_case_2(counts_set, m):
                fail_time += 1
    
    # Calculate experimental probability and return the results, both theoretical and experimental
    if case == "1":
        experiment = accept_time / test_time 
        return {"theorem": at_least_pro.real, "experiment": experiment}
    elif case == "2":
        experiment = fail_time / test_time
        return {"theorem": delta.real, "experiment": experiment}


def start_simulation(circuit_func, copies, *args, include_indices=False):
    """
    Simulate quantum circuits and collect results.

    Parameters:
    - circuit_func: Function to generate quantum circuits.
    - copies: Number of repetitions.
    - args: Additional arguments for the circuit function.
    - include_indices: If True, collect additional indices (only for "random" and "special random" circuit).

    Returns:
    - counts_set: List of counts from simulations.
    - indices_set (optional): List of additional indices for randomly shuffled measurements, if include_indices is True.
    """
    indices_set = [] if include_indices else None
    counts_set = []

    for _ in range(copies):
        if include_indices:
            quantum_circuit, indices = circuit_func(*args)
            indices_set.append(indices)
        else:
            quantum_circuit = circuit_func(*args)
        count = run_circuit(quantum_circuit, num_shot=1, backend='qasm_simulator')
        counts_set.append(count)

    return (counts_set, indices_set) if include_indices else counts_set
