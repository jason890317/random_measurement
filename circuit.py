from numpy.linalg import svd
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm
from scipy.linalg import svd
from qiskit.circuit.library import UnitaryGate
import random
from qiskit import transpile
from qiskit_aer import Aer
from qiskit.visualization import circuit_drawer
# def check_for_rank_one(povm):
#     """
#     function to check if a povm is a rank-1 povm
#     """
#     rank_one = True
#     for p in povm:
#         if np.linalg.matrix_rank(p)!=1:
#             rank_one = False
#             return rank_one
#         else:
#             continue
#     return rank_one
            
# # %%
# def compute_rank_one_unitary(povm, atol=1e-13, rtol=0):
#     """
#     This function computes the unitary that rotates the system to the Hilbert space of the ancilla
#     Input:  POVM ---> a list of the elements of POVM
#     Output: Unitary matrix
#     """
           
#     # check if povm is a rank-1 povm:
#     assert check_for_rank_one(povm), "This is not a rank-1 povm"
#     new_povm = []
#     for p in povm:
#         if np.log2(len(povm))%2==0: #still under investigation
#             w, v = np.linalg.eig(p)
#         else: 
#             w, v = np.linalg.eigh(p) #note the that the eigenvenvector is computer for hermitian eigh
#         for eigenvalue, engenvector in zip(w,v):
#             if np.isclose(np.abs(eigenvalue), 0):
#                 continue
#             else:
#                 new_p = np.sqrt(eigenvalue)*engenvector
#                 new_povm.append(new_p)
#     v = np.vstack(new_povm) # arrange the povm elements to form a matrix of dimension: Row X Col*len(povm)
#     v = np.atleast_2d(v.T) # convert to 2d matrix
    
#     u, s, vh = svd(v)    # apply svd
#     tol = max(atol, rtol * s[0])
#     nnz = (s >= tol).sum()
#     ns = vh[nnz:]         # missing rows of v
    
#     # add the missing rows of v to v
#     V = np.vstack((v, ns)) 
    
    
#     # make the unitary a square matrix of dimension N=2^n where n = int(np.ceil(np.log(V.shape[0])))
#     n = int(np.ceil(np.log2(V.shape[0])))
#     N = 2**n      # dimension of system and ancilla Hilber space
#     r,c = V.shape  
    
#     U = np.eye(N, dtype=complex) # initialize Unitary matrix to the identity. Ensure it is complex
#     U[:r,:c] = V[:r,:c] # assign all the elements of V to the corresponding elements of U
    
#     U = U.conj().T  # Transpose the unitary so that the rows are the povm
    
#     # check for unitarity of U
    
#     # check for unitarity of U
    
#     assert np.allclose(U.T.conj()@U, np.eye(N),atol=1e-13), "Failed to construct U"
    
#     return U
    
# # %%
# # Using the original unitary generator

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
    # print(U.T.conj()@U)
    assert np.allclose(U.T.conj()@U, np.eye(N),atol=1e-07), "Failed to construct U"
    
    return U

# %%
# def rank_one_circuit(povm, state, U):
    
#     # Define the quantum and classical registers
#     dim_system = state.shape[1] # dimension of state
#     num_system_qubit = int(np.ceil(np.log2(dim_system))) # total number of qubits for system

#     system_reg = QuantumRegister(num_system_qubit, name='system') # system register

#     N = U.shape[0] # Dimension of the unitary to be applied to system and ancilla

#     num_ancilla_qubit = int(np.ceil(np.log2(N))) - num_system_qubit # total number of qubits for system

#     ancilla_reg = QuantumRegister(num_ancilla_qubit, name='ancilla') # ancilla register

#     U_gate = UnitaryGate(U, label='U') # unitary gate to be applied between system and ancilla
    
    
#     # create the quantum circuit for the system and ancilla
#     qc = QuantumCircuit(system_reg, ancilla_reg, name='circuit')
#     qc.initialize(state[0],system_reg)

#     # reset ancilla to zero
#     qc.reset(ancilla_reg)

#     # append the unitary gate
#     qc.append(U_gate, range(system_reg.size + ancilla_reg.size))

#     # measure only the ancilliary qubits
#     qc.measure_all()
    
#     return qc

# # %%
# def full_rank_circuit(povm, state, U):
    
#     # Define the quantum and classical registers
#     dim_system = state.shape[1] # dimension of state
#     num_system_qubit = int(np.ceil(np.log2(dim_system))) # total number of qubits for system

#     system_reg = QuantumRegister(num_system_qubit, name='system') # system register

#     N = U.shape[0] # Dimension of the unitary to be applied to system and ancilla

#     num_ancilla_qubit = int(np.ceil(np.log2(N))) - num_system_qubit # total number of qubits for system

#     ancilla_reg = QuantumRegister(num_ancilla_qubit, name='ancilla') # ancilla register

#     classical_reg = ClassicalRegister(num_ancilla_qubit, name='measure') # classical register

#     U_gate = UnitaryGate(U, label='U') # unitary gate to be applied between system and ancilla
    
    
#     # create the quantum circuit for the system and ancilla
#     qc = QuantumCircuit(system_reg, ancilla_reg, classical_reg, name='circuit')
#     qc.initialize(state[0],system_reg)

#     # reset ancilla to zero
#     qc.reset(ancilla_reg)

#     # append the unitary gate
#     qc.append(U_gate, range(system_reg.size + ancilla_reg.size))

#     # measure only the ancilliary qubits
#     qc.measure(ancilla_reg, classical_reg)
    
#     return qc

# # %%
# def construct_quantum_circuit(povm, state):
    
#     # compute unitary matrix
#     if check_for_rank_one(povm):
#         U = compute_rank_one_unitary(povm)
#         qc = rank_one_circuit(povm, state, U)
#     else:
#         U = compute_full_rank_unitary(povm)
#         qc = full_rank_circuit(povm, state, U)
    
#     return qc

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

def construct_blended_circuit(blended_set,state,implete_times):
    
    U_blended=compute_full_rank_unitary(blended_set)
    qc=blended_circuit(blended_set,state,U_blended,implete_times)
    
    return qc

def blended_circuit_inverse(povm, state, U,U_inv,m):
    
    # Define the quantum and classical registers
    dim_system = state.shape[1] # dimension of state
    num_system_qubit = int(np.ceil(np.log2(dim_system))) # total number of qubits for system

    system_reg = QuantumRegister(num_system_qubit, name='system') # system register

    N = U.shape[0] # Dimension of the unitary to be applied to system and ancilla

    num_ancilla_qubit = int(np.ceil(np.log2(N))) - num_system_qubit # total number of qubits for system

    ancilla_reg = QuantumRegister(num_ancilla_qubit, name='ancilla') # ancilla register

    classical_reg = ClassicalRegister(num_ancilla_qubit*m, name='measure') # classical register

    U_gate = UnitaryGate(U, label='U') # unitary gate to be applied between system and ancilla
    U_inv_gate=UnitaryGate(U_inv,label='U_v')
    
    # create the quantum circuit for the system and ancilla
    qc = QuantumCircuit(system_reg, ancilla_reg, classical_reg, name='circuit')
    qc.initialize(state[0],system_reg)

    for i in range(0,m):

        # reset ancilla to zero
        qc.reset(ancilla_reg)

        if i%2==0:
        # # append the unitary gate
            qc.append(U_inv_gate, range(system_reg.size + ancilla_reg.size))
        elif i%2==1:
            qc.append(U_gate, range(system_reg.size + ancilla_reg.size))
        
        
        # measure only the ancilliary qubits
        qc.measure(ancilla_reg, classical_reg[i*num_ancilla_qubit:i*num_ancilla_qubit+num_ancilla_qubit])
    
    return qc

def construct_blended_circuit_inverse(blended_set,blended_set_inverse,state,implete_times):
    
    U_blended_inverse=compute_full_rank_unitary(blended_set_inverse)
    U_blended=compute_full_rank_unitary(blended_set)
    qc=blended_circuit_inverse(blended_set,state,U_blended,U_blended_inverse,implete_times)
    
    return qc

def construct_three_outcome_unitary(blended_set_m):
    
    U=[]
    
    for item in blended_set_m:
        u=compute_full_rank_unitary(item)
        U.append(u)
    
    return U
def construct_blended_three_outcome_circuit(blended_set,state,implete_times,m):
    
    U_set=construct_three_outcome_unitary(blended_set)
    
    dim_system = state.shape[1] # dimension of state
    num_system_qubit = int(np.ceil(np.log2(dim_system))) # total number of qubits for system

    system_reg = QuantumRegister(num_system_qubit, name='system') # system register

    N = U_set[0].shape[0] # Dimension of the unitary to be applied to system and ancilla

    num_ancilla_qubit = int(np.ceil(np.log2(N))) - num_system_qubit # total number of qubits for system

    ancilla_reg = QuantumRegister(num_ancilla_qubit, name='ancilla') # ancilla register

    classical_reg = ClassicalRegister(num_ancilla_qubit*implete_times, name='measure') # classical register

    # U_gate = UnitaryGate(U, label='U') # unitary gate to be applied between system and ancilla
    # U_inv_gate=UnitaryGate(U_inv,label='U_v')
    
    U_gate_set=[]
    
    for item in U_set:
        U_gate_set.append(UnitaryGate(item))
    
    
    # create the quantum circuit for the system and ancilla
    qc = QuantumCircuit(system_reg, ancilla_reg, classical_reg, name='circuit')
    qc.initialize(state[0],system_reg)

    for i in range(0,implete_times):

        # reset ancilla to zero
        qc.reset(ancilla_reg)
        
        # # append the unitary gate
        qc.append(U_gate_set[i], range(system_reg.size + ancilla_reg.size))
    
        # measure only the ancilliary qubits
        qc.measure(ancilla_reg, classical_reg[i*num_ancilla_qubit:i*num_ancilla_qubit+num_ancilla_qubit])
    
    return qc

def test_blended_circuit(qc,num_shot,backend='qasm_simulator'):
    
    backend_options = {
    'max_parallel_threads': 10, # batch 10
    'max_memory_mb': 16384,
    }
    backend = Aer.get_backend(backend)
    backend.set_options(**backend_options)
    result = backend.run(qc,shots=num_shot).result()
    counts = result.get_counts(qc)
    
#    config = backend.configuration()

 #   print("Max memory (MB):", config)
    return counts

def random_sequences_circuit(povm,state,m,pro_h):
    
    dim_system = state.shape[1] # dimension of state
    num_system_qubit = int(np.ceil(np.log2(dim_system))) # total number of qubits for system
    system_reg = QuantumRegister(num_system_qubit, name='system') # system register
    
    
    highest_pro_povm=[]
    roh=np.outer(state,state.conj().T)
    
    
    # for item in povm:
    #     print(np.array(item))
    
    # print("beforo: ")
    # print(povm)
    # print()
    np.random.shuffle(povm)
    # print("after: ")
    # print(povm)
    U=[]
    
    num_ancilla_qubit = 1 # total number of qubits for system

    ancilla_reg = QuantumRegister(num_ancilla_qubit, name='ancilla') # ancilla register

    classical_reg = ClassicalRegister(num_ancilla_qubit*m, name='measure') # classical register

    
    
    # create the quantum circuit for the system and ancilla
    qc = QuantumCircuit(system_reg, ancilla_reg, classical_reg, name='circuit')
    qc.initialize(state[0],system_reg)
    # print("povm: ")
    for p in range(len(povm)):
        atol = 1e-07
        if np.trace(povm[p]@roh) > (pro_h - atol):
            
            highest_pro_povm.append(p)
        
        # print(povm[p])
       
        p_inv=np.eye(dim_system)-povm[p]
        u=compute_full_rank_unitary([povm[p],p_inv])
        # print(U[i])
        # reset ancilla to zero
        qc.reset(ancilla_reg)
        U_gate = UnitaryGate(u, label='U') # unitary gate to be applied between system and ancilla
        # append the unitary gate
        qc.append(U_gate, range(system_reg.size + ancilla_reg.size))

        # measure only the ancilliary qubits
        qc.measure(ancilla_reg, classical_reg[p*num_ancilla_qubit:p*num_ancilla_qubit+num_ancilla_qubit])    

        
        
    
    return (qc,highest_pro_povm)



def test_random_circuit(qc,num_shot,backend='qasm_simulator'):
    
    backend_options = {
    'max_parallel_threads': 10, # batch 10
    'max_memory_mb': 16384,
    
    }
    backend = Aer.get_backend(backend)
    backend.set_options(**backend_options)
    result = backend.run(qc,shots=num_shot, memory=True).result()
    counts = result.get_counts(qc)
    # print(result.get_memory(qc))
#    config = backend.configuration()

 #   print("Max memory (MB):", config)
    return counts