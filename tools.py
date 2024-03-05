import numpy as np
from qiskit import transpile
from qiskit_aer import Aer
import circuit as cir
from itertools import product
from collections import Counter
import matplotlib.pyplot as plt

def print_eigenvalue(povm):
    for item in povm:
        eigenval,_=np.linalg.eigh(item)
        print(eigenval)

def show_probability_povm(povm,roh_0,print_pro=False):
    pro=[]
    for item in povm:
        pro.append(np.trace(item@roh_0))
    if print_pro:
        for item in pro:
            print(item)
    return pro
        
def construct_circuit_and_test(povm,state,num_shot,backend='qasm_simulator'):
    qc=cir.construct_quantum_circuit(povm,state)

    backend = Aer.get_backend('qasm_simulator')
    qc=transpile(qc, backend)
    result = backend.run(qc,shots=num_shot).result()
    counts = result.get_counts(qc)
 
    return counts



def generate_binary_strings(n):
    # Generate all combinations of 0 and 1 of length n
    combinations = product('01', repeat=n)
    
    # Join each combination into a string and create a set of these strings
    binary_strings = {''.join(combination) for combination in combinations}
    
    return binary_strings

def construct_blended_circuit_and_test(blended_set,state,num_shot,implete_times,backend='qasm_simulator'):
    U_blended=cir.compute_full_rank_unitary(blended_set)

    qc=cir.blended_circuit(blended_set,state,U_blended,implete_times)
    backend = Aer.get_backend('statevector_simulator')
    result = backend.run(qc,shots=num_shot).result()
    counts = result.get_counts(qc)
  
    return counts

def resolve_blended_result(counts,m):
    for key in counts.keys():
        raw_result=key
    n= int(np.ceil(np.log2(m)))+1
    result=[raw_result[n*i:(i+1)*n] for i in range(m)]
    result=[ int(item, 2) for item in result]
    number_counts = Counter(result)
 
    return number_counts

def plot_sequential_blended_result(labels,values,m):
    plt.figure(figsize=(20, 6))  # Optional: Adjust the figure size
    plt.bar(labels, values, color='skyblue')  # Create a bar chart

    # Add title and labels to the plot
    plt.title('Frequency of Each E_i')
    plt.xlabel('total number of the Ei :' + str(m+1))
    plt.ylabel('measurement times : '+ str(m))

    # Show the plot
    plt.xticks(range(0, m+1))  # Ensure all numbers are shown on x-axis
    plt.show()