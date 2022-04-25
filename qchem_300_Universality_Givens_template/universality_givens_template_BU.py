#! /usr/bin/python3

import sys
from pennylane import numpy as np
from scipy.optimize import minimize
import pennylane as qml


def givens_rotations(a, b, c, d):
    """Calculates the angles needed for a Givens rotation to out put the state with amplitudes a,b,c and d

    Args:
        - a,b,c,d (float): real numbers which represent the amplitude of the relevant basis states (see problem statement). Assume they are normalized.

    Returns:
        - (list(float)): a list of real numbers ranging in the intervals provided in the challenge statement, which represent the angles in the Givens rotations,
        in order, that must be applied.
    """

    # QHACK #
    DEBUG = True
    num_qubits=6
    dev = qml.device('default.qubit', wires=6)

    @qml.qnode(dev)
    def circuit(x, y, z):
        qml.BasisState(np.array([1, 1, 0, 0, 0, 0]), wires=[i for i in range(6)])
        qml.DoubleExcitation(x, wires=[0, 1, 2, 3])
        qml.DoubleExcitation(y, wires=[0, 1, 4, 5])
        # single excitation controlled on qubit 0
        qml.ctrl(qml.SingleExcitation, control=0)(z, wires=[1, 3])
        return qml.state()

    def cost_fn(param):
        statevector = np.real(circuit(*param))
        target = np.zeros(2**6)
        target[int("110000", 2)] = a
        target[int("001100", 2)] = b
        target[int("000011", 2)] = c
        target[int("100100", 2)] = d
        return np.sum((statevector-target)**2)

    stepsize = 0.5
    from pennylane.optimize import NesterovMomentumOptimizer
    theta = np.array([0.0, 0.0, 0.0], requires_grad=True)

    opt = NesterovMomentumOptimizer(stepsize)
    max_iterations = 500
    conv_tol = 1e-12

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

    return theta
    # QHACK #


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block
    inputs = sys.stdin.read().split(",")
    theta_1, theta_2, theta_3 = givens_rotations(
        float(inputs[0]), float(inputs[1]), float(inputs[2]), float(inputs[3])
    )
    print(*[theta_1, theta_2, theta_3], sep=",")
