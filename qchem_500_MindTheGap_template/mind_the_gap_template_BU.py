import sys
import pennylane as qml
from pennylane import numpy as np
from pennylane import hf


def ground_state_VQE(H):
    """Perform VQE to find the ground state of the H2 Hamiltonian.

    Args:
        - H (qml.Hamiltonian): The Hydrogen (H2) Hamiltonian

    Returns:
        - (float): The ground state energy
        - (np.ndarray): The ground state calculated through your optimization routine
    """

    # QHACK #
    from pennylane.optimize import NesterovMomentumOptimizer, AdamOptimizer

    dev = qml.device("default.qubit", wires=H.wires)
    num_qubits = len(H.wires)

    np.random.seed(42)
    DEBUG = True

    def circuit(param, wires):
        qml.BasisState(np.array([1, 1, 0, 0]), wires=wires)
        qml.DoubleExcitation(param, wires=range(4))

    theta = np.array(0.0, requires_grad=True)

    @qml.qnode(dev)
    def cost_fn(param):
        circuit(param, wires=range(num_qubits))
        return qml.expval(H)

    stepsize = 0.4
    opt = NesterovMomentumOptimizer(stepsize)
    max_iterations = 500
    conv_tol = 1e-8

    energy = [cost_fn(theta)]

    for n in range(max_iterations):
        theta, prev_energy = opt.step_and_cost(cost_fn, theta)
        energy.append(cost_fn(theta))

        conv = np.abs(energy[-1] - prev_energy)

        opt.stepsize *= 0.99

        if DEBUG and n % 20 == 0:
            print('Iteration = {:},  Energy = {:.8f} Ha'.format(n, energy[-1]))

        if conv <= conv_tol:
            break

    return energy[-1], theta
    # QHACK #


def create_H1(ground_state, beta, H):
    """Create the H1 matrix, then use `qml.Hermitian(matrix)` to return an observable-form of H1.

    Args:
        - ground_state (np.ndarray): from the ground state VQE calculation
        - beta (float): the prefactor for the ground state projector term
        - H (qml.Hamiltonian): the result of hf.generate_hamiltonian(mol)()

    Returns:
        - (qml.Observable): The result of qml.Hermitian(H1_matrix)
    """
    dev = qml.device("default.mixed", wires=4)
    @qml.qnode(dev)
    def circuit(param, wires):
        qml.BasisState(np.array([1, 1, 0, 0]), wires=wires)
        qml.DoubleExcitation(param, wires=range(4))
        return qml.state()

    # QHACK #
    GS_matrix = circuit(ground_state, wires=range(4))#np.outer(ground_state, np.conj(ground_state))
    H_matrix = qml.utils.sparse_hamiltonian(H).toarray()
    H1_matrix = H_matrix + beta * GS_matrix

    return qml.Hermitian(H1_matrix, wires=range(4))
    # QHACK #


def excited_state_VQE(H1):
    """Perform VQE using the "excited state" Hamiltonian.

    Args:
        - H1 (qml.Observable): result of create_H1

    Returns:
        - (float): The excited state energy
    """

    # QHACK #
    from pennylane.optimize import NesterovMomentumOptimizer, AdamOptimizer

    dev = qml.device("default.qubit", wires=H1.wires)
    num_qubits = len(H1.wires)

    np.random.seed(42)
    DEBUG = True

    def circuit(param, wires):
        qml.templates.StronglyEntanglingLayers(param, wires=wires)

    n_layers = 3
    theta = np.random.uniform(low=0, high=2*np.pi, size=(n_layers, num_qubits, 3))

    @qml.qnode(dev)
    def overlap(params, wires):
        qml.BasisState(np.array([1, 1, 0, 0]), wires=wires)
        qml.DoubleExcitation(ground_state, wires=range(4))
        qml.adjoint(qml.templates.StronglyEntanglingLayers)(params, wires=range(4))
        return qml.probs([0, 1, 2])

    def cost_fn(params, **kwargs):
        h_cost = qml.ExpvalCost(circuit, H, dev)
        h = h_cost(params, **kwargs)
        o = overlap(params, wires=H.wires)
        return h + 1.5 * o[0]

    stepsize = 0.1
    opt = NesterovMomentumOptimizer(stepsize)
    max_iterations = 500
    conv_tol = 1e-8

    energy = [cost_fn(theta)]

    for n in range(max_iterations):
        theta, prev_energy = opt.step_and_cost(cost_fn, theta)
        energy.append(cost_fn(theta))

        conv = np.abs(energy[-1] - prev_energy)

        opt.stepsize *= 0.99

        if DEBUG and n % 20 == 0:
            print('Iteration = {:},  Energy = {:.8f} Ha'.format(n, energy[-1]))

        if conv <= conv_tol:
            break

    return energy[-1]
    # QHACK #


if __name__ == "__main__":
    coord = float(sys.stdin.read())
    symbols = ["H", "H"]
    geometry = np.array([[0.0, 0.0, -coord], [0.0, 0.0, coord]], requires_grad=False)
    mol = hf.Molecule(symbols, geometry)

    H = hf.generate_hamiltonian(mol)()
    E0, ground_state = ground_state_VQE(H)

    beta = 15.0
    H1 = create_H1(ground_state, beta, H)
    E1 = excited_state_VQE(H1)

    answer = [np.real(E0), E1]
    print(*answer, sep=",")
