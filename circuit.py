from numpy.linalg import svd
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm
from scipy.linalg import svd
from qiskit.circuit.library import UnitaryGate

from qiskit import transpile
from qiskit_aer import Aer

def check_for_rank_one(povm):
    """
    function to check if a povm is a rank-1 povm
    """
    rank_one = True
    for p in povm:
        if np.linalg.matrix_rank(p)!=1:
            rank_one = False
            return rank_one
        else:
            continue
    return rank_one
            
# %%
def compute_rank_one_unitary(povm, atol=1e-13, rtol=0):
    """
    This function computes the unitary that rotates the system to the Hilbert space of the ancilla
    Input:  POVM ---> a list of the elements of POVM
    Output: Unitary matrix
    """
           
    # check if povm is a rank-1 povm:
    assert check_for_rank_one(povm), "This is not a rank-1 povm"
    new_povm = []
    for p in povm:
        if np.log2(len(povm))%2==0: #still under investigation
            w, v = np.linalg.eig(p)
        else: 
            w, v = np.linalg.eigh(p) #note the that the eigenvenvector is computer for hermitian eigh
        for eigenvalue, engenvector in zip(w,v):
            if np.isclose(np.abs(eigenvalue), 0):
                continue
            else:
                new_p = np.sqrt(eigenvalue)*engenvector
                new_povm.append(new_p)
    v = np.vstack(new_povm) # arrange the povm elements to form a matrix of dimension: Row X Col*len(povm)
    v = np.atleast_2d(v.T) # convert to 2d matrix
    
    u, s, vh = svd(v)    # apply svd
    tol = max(atol, rtol * s[0])
    nnz = (s >= tol).sum()
    ns = vh[nnz:]         # missing rows of v
    
    # add the missing rows of v to v
    V = np.vstack((v, ns)) 
    
    
    # make the unitary a square matrix of dimension N=2^n where n = int(np.ceil(np.log(V.shape[0])))
    n = int(np.ceil(np.log2(V.shape[0])))
    N = 2**n      # dimension of system and ancilla Hilber space
    r,c = V.shape  
    
    U = np.eye(N, dtype=complex) # initialize Unitary matrix to the identity. Ensure it is complex
    U[:r,:c] = V[:r,:c] # assign all the elements of V to the corresponding elements of U
    
    U = U.conj().T  # Transpose the unitary so that the rows are the povm
    
    # check for unitarity of U
    assert np.allclose(U.T.conj()@U, np.eye(N),atol=1e-13), "Failed to construct U"
    
    return U
    
# %%
# Using the original unitary generator

def compute_full_rank_unitary(povm, atol=1e-13, rtol=0):
    """
    This function computes the unitary that rotates the system to the Hilbert space of the ancilla
    Input:  POVM ---> a list of the elements of POVM
    Output: Unitary matrix
    """
    
         
    # Here square root of the POVM elements were used as a replacement for the vector that form the povm
    povm = [sqrtm(M)for M in povm]
   
    v = np.hstack(povm) # arrange the povm elements to form a matrix of dimension: Row X Col*len(povm)
    v = np.atleast_2d(v) # convert to 2d matrix
    v=v.astype('complex128')
    u, s, vh = svd(v)    # apply svd
    tol = max(atol, rtol * s[0])
    nnz = (s >= tol).sum()
    ns = vh[nnz:]         # missing rows of v
    
    # add the missing rows of v to v
    V = np.vstack((v, ns)) 
    
    
    # make the unitary a square matrix of dimension N=2^n where n = int(np.ceil(np.log(V.shape[0])))
    n = int(np.ceil(np.log2(V.shape[0])))
    N = 2**n      # dimension of system and ancilla Hilber space
    r,c = V.shape  
    
    U = np.eye(N, dtype=complex) # initialize Unitary matrix to the identity. Ensure it is complex
    U[:r,:c] = V[:r,:c] # assign all the elements of V to the corresponding elements of U
    
    U = U.conj().T  # Transpose the unitary so that the rows are the povm
    
    # check for unitarity of U
    assert np.allclose(U.T.conj()@U, np.eye(N),atol=1e-07), "Failed to construct U"
    
    return U

# %%
def rank_one_circuit(povm, state, U):
    
    # Define the quantum and classical registers
    dim_system = state.shape[1] # dimension of state
    num_system_qubit = int(np.ceil(np.log2(dim_system))) # total number of qubits for system

    system_reg = QuantumRegister(num_system_qubit, name='system') # system register

    N = U.shape[0] # Dimension of the unitary to be applied to system and ancilla

    num_ancilla_qubit = int(np.ceil(np.log2(N))) - num_system_qubit # total number of qubits for system

    ancilla_reg = QuantumRegister(num_ancilla_qubit, name='ancilla') # ancilla register

    U_gate = UnitaryGate(U, label='U') # unitary gate to be applied between system and ancilla
    
    
    # create the quantum circuit for the system and ancilla
    qc = QuantumCircuit(system_reg, ancilla_reg, name='circuit')
    qc.initialize(state[0],system_reg)

    # reset ancilla to zero
    qc.reset(ancilla_reg)

    # append the unitary gate
    qc.append(U_gate, range(system_reg.size + ancilla_reg.size))

    # measure only the ancilliary qubits
    qc.measure_all()
    
    return qc

# %%
def full_rank_circuit(povm, state, U):
    
    # Define the quantum and classical registers
    dim_system = state.shape[1] # dimension of state
    num_system_qubit = int(np.ceil(np.log2(dim_system))) # total number of qubits for system

    system_reg = QuantumRegister(num_system_qubit, name='system') # system register

    N = U.shape[0] # Dimension of the unitary to be applied to system and ancilla

    num_ancilla_qubit = int(np.ceil(np.log2(N))) - num_system_qubit # total number of qubits for system

    ancilla_reg = QuantumRegister(num_ancilla_qubit, name='ancilla') # ancilla register

    classical_reg = ClassicalRegister(num_ancilla_qubit, name='measure') # classical register

    U_gate = UnitaryGate(U, label='U') # unitary gate to be applied between system and ancilla
    
    
    # create the quantum circuit for the system and ancilla
    qc = QuantumCircuit(system_reg, ancilla_reg, classical_reg, name='circuit')
    qc.initialize(state[0],system_reg)

    # reset ancilla to zero
    qc.reset(ancilla_reg)

    # append the unitary gate
    qc.append(U_gate, range(system_reg.size + ancilla_reg.size))

    # measure only the ancilliary qubits
    qc.measure(ancilla_reg, classical_reg)
    
    return qc

# %%
def construct_quantum_circuit(povm, state):
    
    # compute unitary matrix
    if check_for_rank_one(povm):
        U = compute_rank_one_unitary(povm)
        qc = rank_one_circuit(povm, state, U)
    else:
        U = compute_full_rank_unitary(povm)
        qc = full_rank_circuit(povm, state, U)
    
    return qc

def blended_circuit(povm, state, U,m):
    
    # Define the quantum and classical registers
    dim_system = state.shape[1] # dimension of state
    num_system_qubit = int(np.ceil(np.log2(dim_system))) # total number of qubits for system

    system_reg = QuantumRegister(num_system_qubit, name='system') # system register

    N = U.shape[0] # Dimension of the unitary to be applied to system and ancilla

    num_ancilla_qubit = int(np.ceil(np.log2(N))) - num_system_qubit # total number of qubits for system

    ancilla_reg = QuantumRegister(num_ancilla_qubit, name='ancilla') # ancilla register

    classical_reg = ClassicalRegister(num_ancilla_qubit*m, name='measure') # classical register

    U_gate = UnitaryGate(U, label='U') # unitary gate to be applied between system and ancilla
    
    
    # create the quantum circuit for the system and ancilla
    qc = QuantumCircuit(system_reg, ancilla_reg, classical_reg, name='circuit')
    qc.initialize(state[0],system_reg)

    for i in range(0,m):

        # reset ancilla to zero
        qc.reset(ancilla_reg)

        # append the unitary gate
        qc.append(U_gate, range(system_reg.size + ancilla_reg.size))

        # measure only the ancilliary qubits
        qc.measure(ancilla_reg, classical_reg[i*num_ancilla_qubit:i*num_ancilla_qubit+num_ancilla_qubit])
    
    return qc

def construct_circuit_and_test(povm,state,num_shot,backend='qasm_simulator'):
    qc=construct_quantum_circuit(povm,state)

    backend = Aer.get_backend(backend)
    qc=transpile(qc, backend)
    result = backend.run(qc,shots=num_shot).result()
    counts = result.get_counts(qc)
 
    return counts


def construct_blended_circuit(blended_set,state,implete_times):
    
    U_blended=compute_full_rank_unitary(blended_set)
    qc=blended_circuit(blended_set,state,U_blended,implete_times)
    
    return qc

def test_blended_circuit(qc,num_shot,backend='qasm_simulator'):
    
    backend_options = {
    'max_parallel_threads': 10, # batch 10
    'max_memory_mb': 16384,
    }
    backend = Aer.get_backend(backend)
    # backend.set_options(**backend_options)
    result = backend.run(qc,shots=num_shot).result()
    counts = result.get_counts(qc)
#    config = backend.configuration()

 #   print("Max memory (MB):", config)
    return counts
