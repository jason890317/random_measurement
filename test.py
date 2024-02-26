from random_measurement import generate_povm_set_case_1,generate_povm_set_case_2,is_a_proper_povm
from blended_measurement import blended_measurement
from qiskit.extensions import *
from qiskit.circuit.add_control import add_control
from qiskit.visualization import plot_histogram
from qiskit import  BasicAer,transpile, QuantumRegister, ClassicalRegister, QuantumCircuit, execute, BasicAer
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm

n=np.array([[0.07516195+0.00000000e+00j ,0.02756844-2.00136484e-02j],
 [0.02756844+2.00136484e-02j, 0.10055371+6.93889390e-18j]])

eigenval,eigenvec=np.linalg.eigh(sqrtm(n).astype('complex128'))
qc = QuantumCircuit(1, 1)

unitary_gate = UnitaryGate(eigenvec)

#Apply the unitary gate to the qubit
qc.append(unitary_gate, [0])
# Measure the qubit in the computational basis
qc.measure(0, 0)

# Execute the circuit
backend = BasicAer.get_backend('qasm_simulator')
result = execute(qc, backend, shots=10240).result()
counts = result.get_counts(qc)

print(counts)

# Define the state |0>
psi = np.array([1, 0])

# Define the POVM element E
E = np.array([[0.07516195+0.00000000e+00j, 0.02756844-2.00136484e-02j],
              [0.02756844+2.00136484e-02j, 0.10055371+6.93889390e-18j]])

# Calculate the probability P = <psi|E|psi>
P = np.dot(psi.conj().T, np.dot(E, psi))

print("Probability:", P.real)
