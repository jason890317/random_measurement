import numpy as np
from tools import (
    compute_probability_for_povm_set, resolve_blended_result_case_1, resolve_blended_result_case_2,
    resolve_random_result_case_special, resolve_random_result_case_1, resolve_blended_result_case_special,
    resolve_blended_result_case_interweave, generate_permutations, resolve_blended_three_result,
    resolve_random_result_case_2
)
from blended_measurement import (
    blended_measurement, inverse_blended_measurement, optimizing_blended_measurement, three_outcome_blended_measurement
)
from circuit import (
    blended_circuit, random_sequences_circuit, run_circuit,
    interweave_blended_circuit, three_outcome_blended_circuit
)

def run_experiment(circuit_func, counts_set,copies, *args):
    
    for _ in range(copies):
        quantum_circuit = circuit_func(*args)
        count = run_circuit(quantum_circuit, num_shot=1, backend='qasm_simulator')
        counts_set.append(count)
    return counts_set


def quantum_event_identification(copies, d, m, gate_num_times, povm_set, rho, case, state, test_time, case_1_high, method):
    
    # Initialize the count of accept times and the array for storing the eperiment results, count.
    accept_time = 0
    counts_set = []
    
    if method == 'special_random':
        shuffled_indices_set = []
        
        for _ in range(copies):
            quantum_circuit, shuffled_indices = random_sequences_circuit(povm_set, state, m, case_1_high)
            count = run_circuit(quantum_circuit, num_shot=1, backend='qasm_simulator')
            counts_set.append(count)
            shuffled_indices_set.append(shuffled_indices)
        accept_time = resolve_random_result_case_special(counts_set, shuffled_indices_set, m)

    elif method in ['special_blended', 'optimizing_blended']:
        if method == "optimizing_blended":
            blended_set = optimizing_blended_measurement(povm_set, d, m)
        else:
            blended_set = blended_measurement(povm_set, d, m)
        counts_set=run_experiment(blended_circuit,counts_set,copies, blended_set, state, int(gate_num_times * m))
        accept_time = resolve_blended_result_case_special(counts_set, m, gate_num_times)

    elif method == "interweave":
        blended_set = blended_measurement(povm_set, d, m)
        blended_set_inv = inverse_blended_measurement(povm_set, d, m)
        counts_set=run_experiment(interweave_blended_circuit, counts_set,copies, blended_set, blended_set_inv, state, int(gate_num_times * m))
        accept_time = resolve_blended_result_case_interweave(counts_set, m, gate_num_times)

    elif method == "blended_three":
        permutation = generate_permutations(m, int(5 * m))
        three_outcome_blended_set = three_outcome_blended_measurement(povm_set, d, m, permutation)
        counts_set=run_experiment(three_outcome_blended_circuit, counts_set,copies,three_outcome_blended_set, state, int(5 * m))
        accept_time = resolve_blended_three_result(counts_set, m, permutation, int(5 * m))

    experiment = accept_time / m
    
    result = {"theorem": 0, "experiment": experiment}
    
    return result
    

    
def quantum_event_finding(copies, d, m, gate_num_times, povm_set, rho, case, state, test_time, case_1_high, method):
    
    accept_time = 0
    counts_set = []
    
    povm_set_probability = compute_probability_for_povm_set(povm_set, rho, False)
    povm_set_probability = sorted(povm_set_probability, reverse=True)

    if method == 'random':
        if case == 1:
            epsilon = 1 - povm_set_probability[0]
            beta = sum(povm_set_probability[1:])
            at_least_pro = ((1 - epsilon) ** 7) / (1296 * ((1 + beta) ** 7))
        elif case == 2:
            delta = 2 * sum(povm_set_probability)

        for _ in range(test_time):
            quantum_circuit, shuffled_indices = random_sequences_circuit(povm_set, state, m, case_1_high)
            count = run_circuit(quantum_circuit, num_shot=1, backend='qasm_simulator')
            if (case == 1 and resolve_random_result_case_1(count, shuffled_indices)) or \
               (case == 2 and resolve_random_result_case_2(count)):
                accept_time += 1
        experiment = accept_time / test_time

    elif method == 'blended':
        if case == 1:
            epsilon = 1 - povm_set_probability[0]
            beta = sum(povm_set_probability[1:])
            at_least_pro = ((1 - epsilon) ** 3) / (12 * (1 + beta))
        elif case == 2:
            delta = sum(povm_set_probability)

        blended_set = blended_measurement(povm_set, d, m)
        quantum_circuit = blended_circuit(blended_set, state, m)
        
        for _ in range(int(test_time / 50)):
            count = run_circuit(quantum_circuit, 50)
            counts_set.extend(count.items())

        count_set_dict = {}
        for key, count in counts_set:
            count_set_dict[key] = count_set_dict.get(key, 0) + count

        if case == 1:
            accept_time = resolve_blended_result_case_1(count_set_dict, m)
        elif case == 2:
            accept_time = resolve_blended_result_case_2(count_set_dict, m)
        experiment = accept_time / test_time

    if case == 1 and method in ["blended", "random"]:
        result = {"theorem": at_least_pro.real, "experiment": experiment}
    elif case == 2 and method in ["blended", "random"]:
        result = {"theorem": delta.real, "experiment": experiment}
   
    
    return result
