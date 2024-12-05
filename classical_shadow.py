import numpy as np
from tools import top_half_indices, split_shadow_median
from scipy.stats import unitary_group
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.circuit.library import UnitaryGate
from circuit import run_circuit

def classical_shadow(copies, d, m, povm_set, state):
    """
    Compute the classical shadow and evaluate performance based on POVM sets.

    Parameters:
    - copies: Number of samples used.
    - d: Dimension of the quantum system.
    - m: Number of POVM elements.
    - povm_set: List of POVM elements.
    - state: Initial quantum state.

    Returns:
    - result (dict): A dictionary containing theoretical and experimental results.
    """
    
    # Number of system qubits
    num_system_qubit = int(np.ceil(np.log2(d)))  
    
    # generate Random unitary matrices set
    Haar_random_unitary_set = [unitary_group.rvs(d) for _ in range(copies)]  
    
    measure_outcomes = []  # Store measurement outcomes
    classical_shadow_set = []  # Store shadow snapshots

    for U in Haar_random_unitary_set:
        
        # Create and initialize the quantum circuit
        system_reg = QuantumRegister(num_system_qubit, name='system')
        classical_reg = ClassicalRegister(num_system_qubit, name='measure')
        quantum_circuit = QuantumCircuit(system_reg, classical_reg, name='circuit')
        quantum_circuit.initialize(state[0], system_reg)
        quantum_circuit.append(UnitaryGate(U, label='U'), range(system_reg.size))
        quantum_circuit.measure(system_reg, classical_reg)

        # Run the circuit and record the measurement outcome
        count = run_circuit(quantum_circuit, 1)
        
        # Get the outcome and turn the outcome into a vector
        measure_outcome = int(list(count)[0], 2)
        outcome_vector = np.zeros(d)
        outcome_vector[measure_outcome] = 1
        
        # Append the outcome vector to the list
        measure_outcomes.append(outcome_vector)

    # Construct the classical shadow by the procedure provided in "Predicting Many Properties of a Quantum System from Very Few Measurements"
    for i, U in enumerate(Haar_random_unitary_set):
        outcome_matrix = np.outer(measure_outcomes[i], measure_outcomes[i].T.conj())
        snapshot = ((2**num_system_qubit) + 1) * (U.conj().T @ outcome_matrix @ U) - np.eye(d)
        classical_shadow_set.append(snapshot)

    # Split shadow and evaluate the results based on the median method in "Predicting Many Properties of a Quantum System from Very Few Measurements"
    output = split_shadow_median(povm_set, classical_shadow_set, 1)
    correct = [0 if i < m / 2 else 1 for i in range(m)]
    check = [1 if idx in top_half_indices(output) else 0 for idx in range(m)]

    # Calculate acceptance with the prediction by classical shadow and the correct answer
    xor_result = [a ^ b for a, b in zip(check, correct)]
    accept_time = xor_result.count(0)

    return {"theorem": 0, "experiment": accept_time / m}
