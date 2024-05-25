import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from qiskit.visualization import plot_bloch_multivector
from qiskit.quantum_info import Statevector

# Ensure the necessary libraries are installed
try:
    import qiskit
except ImportError:
    st.error("Qiskit is not installed. Please install Qiskit using `pip install qiskit`.")

# Define the basis states
basis_states = [np.array([1, 0]), np.array([0, 1])]

# Define coefficients for the Perfected Quantum State
c_i = np.array([1/np.sqrt(2), 1/np.sqrt(2)])

# Define the Perfected Quantum State
Psi_perf = sum(c_i[i] * basis_states[i] for i in range(len(c_i)))

# Function to simulate the quantum state
def simulate_quantum_state(state):
    coherence = np.dot(state, state)
    purity = np.linalg.norm(state)**2
    return coherence, purity

# Run the simulation
coherence, purity = simulate_quantum_state(Psi_perf)

# Streamlit app
st.title("Perfected Quantum State Simulation")

st.write("### Simulation Results")
st.write(f"Coherence: {coherence}")
st.write(f"Purity: {purity}")

# Visualizing the state
st.write("### Quantum State Vector")
st.write(Psi_perf)

# Plotting the coefficients of the quantum state
fig, ax = plt.subplots()
ax.bar(['c0', 'c1'], np.abs(c_i)**2)
ax.set_ylabel('Probability Amplitude')
ax.set_title('Quantum State Coefficients')
st.pyplot(fig)

# Convert the state to a Qiskit Statevector
statevector = Statevector(Psi_perf)

# Plot the quantum state on the Bloch Sphere using Qiskit's built-in function
st.write("### Bloch Sphere Representation")
fig2 = plt.figure()
plot_bloch_multivector(statevector)
st.pyplot(fig2)
