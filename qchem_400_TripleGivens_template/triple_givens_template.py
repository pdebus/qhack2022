import sys
import pennylane as qml
from pennylane import numpy as np

NUM_WIRES = 6


def triple_excitation_matrix(gamma):
    """The matrix representation of a triple-excitation Givens rotation.

    Args:
        - gamma (float): The angle of rotation

    Returns:
        - (np.ndarray): The matrix representation of a triple-excitation
    """

    # QHACK #
    N = 2 ** 6
    state1 = int("111000", 2)
    state2 = int("000111", 2)

    diagonal = np.ones(N)
    diagonal[state1] = np.cos(0.5 * gamma)
    diagonal[state2] = np.cos(0.5 * gamma)

    G = np.diag(diagonal)
    G[state1, state2] = -np.sin(0.5 * gamma)
    G[state2, state1] = np.sin(0.5 * gamma)

    return G
    # QHACK #


dev = qml.device("default.qubit", wires=6)


@qml.qnode(dev)
def circuit(angles):
    """Prepares the quantum state in the problem statement and returns qml.probs

    Args:
        - angles (list(float)): The relevant angles in the problem statement in this order:
        [alpha, beta, gamma]

    Returns:
        - (np.tensor): The probability of each computational basis state
    """

    # QHACK #
    G3 = triple_excitation_matrix(angles[2])

    qml.BasisState(np.array([1, 1, 1, 0, 0, 0]), wires=range(NUM_WIRES))
    qml.SingleExcitation(angles[0], wires=[0, 5])
    qml.DoubleExcitation(angles[1], wires=[0, 1, 4, 5])

    qml.QubitUnitary(G3, wires=range(NUM_WIRES))
    # QHACK #

    return qml.probs(wires=range(NUM_WIRES))


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block
    inputs = np.array(sys.stdin.read().split(","), dtype=float)
    probs = circuit(inputs).round(6)
    print(*probs, sep=",")
