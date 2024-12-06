from numpy.linalg import svd
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
import numpy as np
from scipy.linalg import sqrtm
from qiskit.circuit.library import UnitaryGate
from qiskit_aer import Aer

def compute_full_rank_unitary(povm, atol=1e-13, rtol=0):
    """
    Compute a full-rank unitary matrix that maps a POVM into an extended Hilbert space.

    Parameters:
    - povm: List of POVM elements (positive semidefinite matrices).
    - atol, rtol: Tolerances for singular value thresholding.

    Returns:
    - Unitary matrix (complex-valued).
    """
    # Compute the square root of each POVM element
    povm = [sqrtm(M) for M in povm]
    
    # Combine POVM elements into a single matrix
    v = np.hstack(povm).astype('complex128')
    v = np.atleast_2d(v)  # Ensure 2D matrix
    
    # Perform Singular Value Decomposition (SVD)
    u, s, vh = svd(v)
    
    # Threshold singular values to determine null space
    tol = max(atol, rtol * s[0])
    nnz = (s >= tol).sum()
    ns = vh[nnz:]  # Null space rows
    
    # Complete the matrix with the null space
    V = np.vstack((v, ns))
    
    # Determine size of extended Hilbert space
    n = int(np.ceil(np.log2(V.shape[0])))
    dim_of_unitary = 2**n  # Next power of 2
    
    # Embed V into an identity matrix of the extended space
    unitary = np.eye(dim_of_unitary, dtype=complex)
    unitary[:V.shape[0], :V.shape[1]] = V
    unitary = unitary.conj().T  # Transpose for proper embedding
    
    # Verify that unitary is unitary
    # assert np.allclose(unitary.T.conj() @ unitary, np.eye(dim_of_unitary), atol=1e-8), "Failed to construct unitary"
    
    return unitary


def initialize_quantum_circuit(initial_state, dim_of_unitary,measurement_implemented_time):
    """
    Initialize a quantum circuit with system, ancilla, and classical registers.

    Parameters:
    - initial_state: Initial quantum state (array or list).
    - unitary: Optional unitary matrix to convert into a gate.
    - inverse_unitary: Optional inverse unitary matrix to convert into a gate.
    - measurement_implemented_time: Number of repetitions for the measurement process (default: 1).

    Returns:
    - QuantumCircuit: The initialized quantum circuit.
    - system_reg: System quantum register.
    - ancilla_reg: Ancilla quantum register.
    - classical_reg: Classical register for measurements.
    """
    # Determine the number of qubits for the system and ancilla
    dim_system = initial_state.shape[1]
    num_system_qubit = int(np.ceil(np.log2(dim_system)))  # System qubits
    num_ancilla_qubit = int(np.ceil(np.log2(dim_of_unitary))) - num_system_qubit  # Ancilla qubits

    # Initialize quantum and classical registers
    system_reg = QuantumRegister(num_system_qubit, name='system')
    ancilla_reg = QuantumRegister(num_ancilla_qubit, name='ancilla')
    classical_reg = ClassicalRegister(num_ancilla_qubit * measurement_implemented_time, name='measure')

    # Create the initial quantum circuit
    initial_quantum_circuit = QuantumCircuit(system_reg, ancilla_reg, classical_reg, name='circuit')
    initial_quantum_circuit.initialize(initial_state[0], system_reg)  # Initialize the system initial_state

    return initial_quantum_circuit, system_reg, ancilla_reg, classical_reg, num_ancilla_qubit




def construct_blended_circuit(initial_state, unitary, measurement_implemented_time):
    """
    Construct a quantum circuit for the blended measurement procedure.

    Parameters:
    - initial_state: Initial quantum state.
    - unitary: Unitary matrix embedding the POVM.
    - measurement_implemented_time: Number of repetitions of the measurement process.

    Returns:
    - quantum circuit: The constructed quantum circuit.
    """
    
    # Dimension of the unitary matrix
    dim_of_unitary = unitary.shape[0]  
    
    
    # initialize a quantum circuit with system, ancilla, and classical registers
    quantum_circuit, system_reg, ancilla_reg, classical_reg ,num_ancilla_qubit= initialize_quantum_circuit(initial_state,dim_of_unitary, measurement_implemented_time)


    # Make unitary into quantum gate
    unitary_quantum_gate = UnitaryGate(unitary, label='unitary')
    quantum_circuit = QuantumCircuit(system_reg, ancilla_reg, classical_reg, name='circuit')
   

    # Add the unitary operation and measurement measurement_implemented_time times
    for i in range(measurement_implemented_time):
        quantum_circuit.reset(ancilla_reg)  # Reset ancilla qubits
        quantum_circuit.append(unitary_quantum_gate, range(system_reg.size + ancilla_reg.size))  # Apply unitary
        quantum_circuit.measure(ancilla_reg, classical_reg[i * num_ancilla_qubit:(i + 1) * num_ancilla_qubit])  # Measure ancilla

    return quantum_circuit

def blended_circuit(blended_set, initial_state, measurement_implemented_time):
    """
    Create a blended measurement circuit for a given POVM set.

    Parameters:
    - blended_set: List of POVM elements.
    - initial_state: Initial quantum state.
    - measurement_implemented_time: Number of repetitions for the measurement process.

    Returns:
    - QuantumCircuit: The blended measurement quantum circuit.
    """
    # Compute the unitary matrix for the blended POVM set
    unitary_of_blended_set = compute_full_rank_unitary(blended_set)
    
    # Construct the quantum circuit using the unitary matrix
    return construct_blended_circuit(initial_state, unitary_of_blended_set, measurement_implemented_time)


def construct_interweave_blended_circuit(initial_state, unitary, inverse_unitary, measurement_implemented_time):
    """
    Construct a quantum circuit for the interweave blended measurement procedure.

    Parameters:
    - initial_state: Initial quantum initial_state (array or list).
    - unitary: Unitary matrix for the normal operation.
    - inverse_unitary: Unitary matrix for the inverse operation.
    - measurement_implemented_time: Number of repetitions of the measurement process.

    Returns:
    - QuantumCircuit: The constructed interweave blended quantum circuit.
    """
    
    # Dimension of the unitary matrix
    dim_of_unitary = unitary.shape[0]  
    
    # Initialize a quantum circuit with system, ancilla, and classical registers
    quantum_circuit, system_reg, ancilla_reg, classical_reg ,  num_ancilla_qubit= initialize_quantum_circuit(initial_state,dim_of_unitary, measurement_implemented_time)
    
    # Make unitary into quantum gates
    unitary_quantum_gate = UnitaryGate(unitary, label='unitary')
    inverse_unitary_quantum_gate = UnitaryGate(inverse_unitary, label='inverse_unitary')
   

    # Interweave normal and inverse operations and measure ancilla
    for i in range(measurement_implemented_time):
        quantum_circuit.reset(ancilla_reg)  # Reset ancilla qubits
        # Apply inverse_unitary for even iterations, unitary for odd iterations
        quantum_circuit.append(inverse_unitary_quantum_gate if i % 2 == 0 else unitary_quantum_gate, range(system_reg.size + ancilla_reg.size))
        quantum_circuit.measure(ancilla_reg, classical_reg[i * num_ancilla_qubit:(i + 1) * num_ancilla_qubit])  # Measure ancilla

    return quantum_circuit

def interweave_blended_circuit(blended_set, inverse_blended_set, initial_state, measurement_implemented_time):
    """
    Create a circuit for the interweave blended measurement procedure.

    Parameters:
    - blended_set: List of POVM elements for the forward operation.
    - inverse_blended_set: List of POVM elements.
    - initial_state: Initial quantum initial_state.
    - measurement_implemented_time: Number of repetitions for the measurement process.

    Returns:
    - QuantumCircuit: The interwoven blended quantum circuit.
    """
    # Compute unitary matrices for normal and inverse blended sets
    unitary_of_inverse_blended_set = compute_full_rank_unitary(inverse_blended_set)
    unitary_of_blended_set = compute_full_rank_unitary(blended_set)
    
    # Construct the interweave blended circuit
    return construct_interweave_blended_circuit(initial_state, unitary_of_blended_set, unitary_of_inverse_blended_set, measurement_implemented_time)


def three_outcome_blended_circuit(blended_set, initial_state, measurement_implemented_time):
    """
    Create a circuit for the three-outcome blended measurement procedure.

    Parameters:
    - blended_set: List of POVM elements.
    - initial_state: Initial quantum initial_state.
    - measurement_implemented_time: Number of repetitions for the measurement process.

    Returns:
    - QuantumCircuit: The constructed three-outcome blended quantum circuit.
    """
    # Compute unitary matrices for three-outcome blended set
    unitary_of_three_outcome_bleneded_set = [compute_full_rank_unitary(item) for item in blended_set]
    
    # Construct the quantum circuit for the three-outcome blended measurement
    return construct_three_outcome_blended_circuit(initial_state, unitary_of_three_outcome_bleneded_set, measurement_implemented_time)


def construct_three_outcome_blended_circuit(initial_state, unitary, measurement_implemented_time):
    """
    Construct the three-outcome blended measurement quantum circuit.

    Parameters:
    - initial_state: Initial quantum initial_state.
    - unitary: List of unitary matrices corresponding to the three-outcome measurement.
    - measurement_implemented_time: Number of repetitions for the measurement process.

    Returns:
    - QuantumCircuit: The constructed quantum circuit.
    """

    # Dimension of the unitary matrix
    dim_of_unitary = unitary[0].shape[0]  

    
    # initialize a quantum circuit with system, ancilla, and classical registers
    quantum_circuit, system_reg, ancilla_reg, classical_reg ,num_ancilla_qubit= initialize_quantum_circuit(initial_state,dim_of_unitary, measurement_implemented_time)
   

    # Make unitary into quantum gates
    unitary_quantum_gate = [UnitaryGate(u, label='unitary') for u in unitary]

    # Add unitary operations and measurements for each repetition
    for i in range(measurement_implemented_time):
        quantum_circuit.reset(ancilla_reg)  # Reset ancilla qubits
        quantum_circuit.append(unitary_quantum_gate[i], range(system_reg.size + ancilla_reg.size))  # Apply unitary
        quantum_circuit.measure(ancilla_reg, classical_reg[i * num_ancilla_qubit:(i + 1) * num_ancilla_qubit])  # Measure ancilla

    return quantum_circuit




def random_sequences_circuit(povm, initial_state, measurement_implemented_time, pro_h):
    """
    Construct a quantum circuit for the random sequences measurement procedure.

    Parameters:
    - povm: List of POVM elements.
    - initial_state: Initial quantum initial_state.
    - measurement_implemented_time: Number of measurements.
    - pro_h: Probability threshold for selecting high-probability POVM elements.

    Returns:
    - QuantumCircuit: The constructed quantum circuit.
    - measurements_shuffled_indices: Shuffled measurements_shuffled_indices of the POVM elements.
    """

    
    # Determine the number of qubits for the system
    dim_system = initial_state.shape[1]
    num_system_qubit = int(np.ceil(np.log2(dim_system)))  # System qubits

    # Initialize quantum and classical registers
    system_reg = QuantumRegister(num_system_qubit, name='system')
    num_ancilla_qubit = 1  # Single qubit for ancilla
    ancilla_reg = QuantumRegister(num_ancilla_qubit, name='ancilla')
    classical_reg = ClassicalRegister(num_ancilla_qubit * measurement_implemented_time, name='measure')

    # Create quantum circuit and initialize the system initial_state
    quantum_circuit = QuantumCircuit(system_reg, ancilla_reg, classical_reg, name='circuit')
    quantum_circuit.initialize(initial_state[0], system_reg)

    # Compute the density matrix of the initial_state
    roh = np.outer(initial_state, initial_state.conj().T)

    # Shuffle POVM measurements_shuffled_indices for randomness
    measurements_shuffled_indices = np.arange(len(povm))
    np.random.shuffle(measurements_shuffled_indices)
    povm = povm[measurements_shuffled_indices]  # Shuffle POVM elements

    # Track POVM elements with probabilities exceeding the threshold
    highest_pro_povm = []

    # Iterate through shuffled POVM elements
    for p in range(len(povm)):
        # Check if the POVM element's trace with the density matrix exceeds the threshold
        if np.trace(povm[p] @ roh) > (pro_h - 1e-07):
            highest_pro_povm.append(p)

        # Compute the complementary operator and construct a unitary
        p_inv = np.eye(dim_system) - povm[p]
        u = compute_full_rank_unitary([povm[p], p_inv])

        # Reset ancilla qubits
        quantum_circuit.reset(ancilla_reg)

        # Apply the unitary operation
        unitary_quantum_gate = UnitaryGate(u, label='unitary')
        quantum_circuit.append(unitary_quantum_gate, range(system_reg.size + ancilla_reg.size))

        # Measure ancilla qubits
        quantum_circuit.measure(ancilla_reg, classical_reg[p * num_ancilla_qubit:(p + 1) * num_ancilla_qubit])
    
    # Also return the measurements_shuffled_indices of the highest probability POVM elements for further count the accepting probability in tool.py
    return quantum_circuit, measurements_shuffled_indices



def run_circuit(quantum_circuit, num_shot, backend='qasm_simulator', memory=False):
    """
    Simulate a quantum circuit and retrieve measurement counts.

    Parameters:
    - quantum_circuit: The quantum circuit to simulate.
    - num_shot: Number of shots (simulations) to perform.
    - backend: Qiskit backend for simulation (default: 'qasm_simulator').
    - memory: If True, retrieves the individual measurement outcomes for each shot.

    Returns:
    - dict: Measurement counts from the simulation.
    """
    # Configure backend simulation options
    backend_options = {
        'max_parallel_threads': 10,  # Maximum parallel threads for efficiency
        'max_memory_mb': 16384,  # Allocate maximum memory
    }
    backend_instance = Aer.get_backend(backend)  # Get the specified backend
    backend_instance.set_options(**backend_options)  # Apply backend options

    # Run the simulation
    result = backend_instance.run(quantum_circuit, shots=num_shot, memory=memory).result()

    # Return the measurement counts
    return result.get_counts(quantum_circuit)
