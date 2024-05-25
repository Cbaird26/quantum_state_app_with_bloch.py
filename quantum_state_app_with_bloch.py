import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from qiskit.visualization import plot_bloch_multivector
from qiskit.quantum_info import Statevector

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

# Create a figure for the Bloch Sphere visualization
fig2, ax2 = plt.subplots(subplot_kw={'projection': '3d'})
plot_bloch_multivector(statevector, title="Bloch Sphere Representation", ax=ax2)
st.pyplot(fig2)
