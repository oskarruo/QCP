def nearest_neighbour_IQP_ansatz(n_qubits):
    """
    Nearest-neighbour IQP ansatz.

    Params:
        n_qubits: number of qubits in the circuit

    Returns:
        gates: list of gates in IQP format [[[i,j]], ...]
    """
    gates = []

    # Single-qubit Z terms
    for i in range(n_qubits):
        gates.append([[i]])

    # Nearest-neighbour ZZ interactions
    for i in range(n_qubits - 1):
        gates.append([[i, i + 1]])

    return gates


def fully_connected_IQP_ansatz(n_qubits):
    """
    Fully-connected IQP ansatz.

    Params:
        n_qubits: number of qubits in the circuit

    Returns:
        gates: list of gates in IQP format [[[i,j]], ...]
    """
    gates = []

    # Single-qubit Z terms
    for i in range(n_qubits):
        gates.append([[i]])

    # Fully-connected ZZ interactions
    for i in range(n_qubits):
        for j in range(i + 1, n_qubits):
            gates.append([[i, j]])

    return gates
