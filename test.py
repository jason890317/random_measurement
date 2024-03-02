from qiskit import QuantumCircuit, Aer, transpile
from qiskit.visualization import plot_histogram

# Create a quantum circuit
qc = QuantumCircuit(1, 1)
qc.h(0)
qc.measure(0, 0)
qc.save_statevector()

# Simulate the circuit
simulator = Aer.get_backend('aer_simulator')
tqc = transpile(qc, simulator)
result = simulator.run(tqc).result()

# Get the statevector
statevector = result.get_statevector(qc)

# Visualize the result
counts = result.get_counts(qc)
plot_histogram(counts)
