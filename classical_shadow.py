
import numpy as np

from tools import show_probability_povm,top_half_indices,split_shadow_median
from scipy.stats import unitary_group
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
import numpy as np
from qiskit.circuit.library import UnitaryGate
from circuit import test_blended_circuit

def classical_shadow(copies,d,m,povm_set,state):
  
    # print(state)
    U_set=[]
    measure_outcome_set=[]
    classical_shadow_set=[]
    
    num_system_qubit = int(np.ceil(np.log2(d)))
    
    for i in range(copies):
        U=unitary_group.rvs(d)
        U_set.append(U)
        # print(U)
    for i in range(copies):
        
        system_reg = QuantumRegister(num_system_qubit, name='system') 
        classical_reg = ClassicalRegister(num_system_qubit, name='measure')
        qc= QuantumCircuit(system_reg, classical_reg, name='circuit')
        # print(state)
        qc.initialize(state[0],system_reg)
        # print(U_set[i])
        U_gate = UnitaryGate(U_set[i], label='U')
        qc.append(U_gate, range(system_reg.size))
        qc.measure(system_reg, classical_reg[:])
        count=test_blended_circuit(qc,1)
        measure_outcome=int(list(count)[0],2)
        measure_outcome_bit_str=np.zeros(d)
        measure_outcome_bit_str[measure_outcome]=1
        # print(measure_outcome)
        # print(measure_outcome_bit_str)
        measure_outcome_set.append(measure_outcome_bit_str)
        
    for i in range(copies):
        
        outcome_matrix=np.outer(measure_outcome_set[i],measure_outcome_set[i].T.conj())
    
        snapshot=((2**num_system_qubit)+1)*(U_set[i].conj().T@outcome_matrix@U_set[i])-np.eye(d)
        # snapshot=(U_set[i].conj().T@outcome_matrix@U_set[i])
        classical_shadow_set.append(snapshot)
    # print(classical_shadow_set)
    # print(np.trace(classical_shadow_set[i]) for i in range(len(classical_shadow_set)))
    # predicted_state=np.mean(classical_shadow_set,axis=0)
    # print(np.trace(predicted_state))
    # show_probability_povm(measurements,predicted_state,True)
    # show_probability_povm(measurements,roh,True)
    
    k=10
    
    output=split_shadow_median(povm_set, classical_shadow_set,k)
    
    
    correct = [0 if i <m/2 else 1 for i in range(m)]
    check=[0 for _ in range(m)]
    
    # for idx in top_half_indices(output):
    #     if idx>=int(m/2):
    #         accept+=1
    for idx in top_half_indices(output):
        check[idx]=1
    
    print(correct)
    print(check)
    xor_result = [a ^ b for a, b in zip(check, correct)]
    accept_time=xor_result.count(0)
    
    result={"theorem":0,"experiment":accept_time/int(m)}
    
    return result