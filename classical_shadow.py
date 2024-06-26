
import numpy as np
from generate_povm import  generate_povm_epson_case_special
from tools import show_probability_povm,top_half_indices
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
import numpy as np
from qiskit.circuit.library import UnitaryGate
from circuit import test_blended_circuit

def random_unitary(d):
    
    real_part = np.random.rand(d, d)
    imaginary_part = np.random.rand(d, d)
    A= real_part + 1j * imaginary_part
    U, _ = np.linalg.qr(A)
    # print(U)
    return U


def split_shadow_median(measurements, classical_shadow_set,K):
    
    # print(classical_shadow_set)
    N=len(classical_shadow_set)
    splited_shadow=np.array_split(classical_shadow_set, K)
    # print(splited_shadow)
    
    means_splited_shadow=[]
    
    for item in splited_shadow:
        sum=0
        for copy in item:
            sum+=copy
        # print(sum)
        mean=sum/(int(N/K))
        # print(mean)
        means_splited_shadow.append(mean)
    
    # for item in means_splited_shadow:
    
    #     print("trace :"+str(np.trace(abs(item))))
    probability_set_all_shadow=[[] for i in range(len(measurements)) ]
    
    for i in range(len(measurements)):
        for j in range(K):
            probability_set_all_shadow[i].append(abs(np.trace(measurements[i]@means_splited_shadow[j])))
    
    output=[]
    
    for item in probability_set_all_shadow:
        # print(item)
        median=np.median(item)
        # print(median)
        indices = np.where(item == median)[0]
        output.append(median)
    for item in output:
        print(item)
        
    return output

if __name__ == '__main__':
    
    dimension=4
    measurement_number=20
    rank=2
    pro_h=0.9
    pro_l=0.001
    
    copies=1
    state=np.array([np.hstack((1,np.zeros(dimension-1)))])[0]   
    roh=np.outer(state,state.T.conj())
    # print(state)
    U_set=[]
    measure_outcome_set=[]
    classical_shadow_set=[]
    
    measurements=generate_povm_epson_case_special(dimension,measurement_number,rank,pro_h,pro_l,roh)
    
    num_system_qubit = int(np.ceil(np.log2(dimension)))
    
    for i in range(copies):
        U=random_unitary(dimension)
        U_set.append(U)
        # print(U)
    for i in range(copies):
        
        system_reg = QuantumRegister(num_system_qubit, name='system') 
        classical_reg = ClassicalRegister(num_system_qubit, name='measure')
        qc= QuantumCircuit(system_reg, classical_reg, name='circuit')
        qc.initialize(state,system_reg)
        # print(U_set[i])
        U_gate = UnitaryGate(U_set[i], label='U')
        qc.append(U_gate, range(system_reg.size))
        qc.measure(system_reg, classical_reg[:])
        count=test_blended_circuit(qc,1)
        measure_outcome=int(list(count)[0][::-1],2)
        measure_outcome_bit_str=np.zeros(dimension)
        measure_outcome_bit_str[measure_outcome]=1
        # print(count)
        # print(measure_outcome_bit_str)
        measure_outcome_set.append(measure_outcome_bit_str)
        
    for i in range(copies):
        
        outcome_matrix=np.outer(measure_outcome_set[i].T.conj(),measure_outcome_set[i])
    
        snapshot=((2**num_system_qubit)+1)*(U_set[i].conj().T@outcome_matrix@U_set[i])-np.eye(dimension)
        # snapshot=(U_set[i].conj().T@outcome_matrix@U_set[i])
        classical_shadow_set.append(snapshot)
    # print(classical_shadow_set)
    # print(np.trace(classical_shadow_set[i]) for i in range(len(classical_shadow_set)))
    # predicted_state=np.mean(classical_shadow_set,axis=0)
    # print(np.trace(predicted_state))
    # show_probability_povm(measurements,predicted_state,True)
    # show_probability_povm(measurements,roh,True)
    output=split_shadow_median(measurements, classical_shadow_set,1)
    accept=0
    for idx in top_half_indices(output):
        if idx>=int(measurement_number/2):
            accept+=1
            
    print("sucess probabiliy: "+str(accept/int(measurement_number/2)))