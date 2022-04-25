#! /usr/bin/python3

import sys
import pennylane as qml
from pennylane import numpy as np


dev = qml.device("default.qubit", wires=2)

def prepare_entangled(alpha, beta):
    """Construct a circuit that prepares the (not necessarily maximally) entangled state in terms of alpha and beta
    Do not forget to normalize.

    Args:
        - alpha (float): real coefficient of |00>
        - beta (float): real coefficient of |11>
    """

    # QHACK #
    norm = np.sqrt(alpha ** 2 + beta ** 2)
    qml.RY(2 * np.arccos(alpha / norm), wires=0)
    qml.CNOT(wires=[0, 1])
    # QHACK #

@qml.qnode(dev)
def chsh_circuit(theta_A0, theta_A1, theta_B0, theta_B1, x, y, alpha, beta):
    """Construct a circuit that implements Alice's and Bob's measurements in the rotated bases

    Args:
        - theta_A0 (float): angle that Alice chooses when she receives x=0
        - theta_A1 (float): angle that Alice chooses when she receives x=1
        - theta_B0 (float): angle that Bob chooses when he receives x=0
        - theta_B1 (float): angle that Bob chooses when he receives x=1
        - x (int): bit received by Alice
        - y (int): bit received by Bob
        - alpha (float): real coefficient of |00>
        - beta (float): real coefficient of |11>

    Returns:
        - (np.tensor): Probabilities of each basis state
    """

    prepare_entangled(alpha, beta)

    # QHACK #
    if x == 0:
        qml.RY(2 * theta_A0, wires=0)
    elif x == 1:
        qml.RY(2 * theta_A1, wires=0)
    else:
        raise ValueError

    if y == 0:
        qml.RY(2 * theta_B0, wires=1)
    elif y == 1:
        qml.RY(2 * theta_B1, wires=1)
    else:
        raise ValueError

    # QHACK #

    return qml.probs(wires=[0, 1])
    

def winning_prob(params, alpha, beta):
    """Define a function that returns the probability of Alice and Bob winning the game.

    Args:
        - params (list(float)): List containing [theta_A0,theta_A1,theta_B0,theta_B1]
        - alpha (float): real coefficient of |00>
        - beta (float): real coefficient of |11>

    Returns:
        - (float): Probability of winning the game
    """

    # QHACK #
    total_prob = 0
    for x in [0, 1]:
        for y in [0, 1]:
            p = chsh_circuit(*params, x, y, alpha, beta)
            #print(x,y,p)
            if x * y == 0:
                total_prob += 0.25 * (p[0] + p[3])
            else:
                total_prob += 0.25 * (p[1] + p[2])

    return total_prob
    # QHACK #
    

def optimize(alpha, beta):
    """Define a function that optimizes theta_A0, theta_A1, theta_B0, theta_B1 to maximize the probability of winning the game

    Args:
        - alpha (float): real coefficient of |00>
        - beta (float): real coefficient of |11>

    Returns:
        - (float): Probability of winning
    """

    def cost(params):
        """Define a cost function that only depends on params, given alpha and beta fixed"""
        return (1 - winning_prob(params, alpha, beta)) ** 2
    # QHACK #

    #Initialize parameters, choose an optimization method and number of steps
    from numpy import random
    from pennylane.optimize import NesterovMomentumOptimizer
    random.seed(100)

    init_params = random.uniform(-np.pi, np.pi, 4)
    opt = NesterovMomentumOptimizer(stepsize=0.1)
    steps = 400
    DEBUG = True

    # QHACK #
    from scipy.optimize import minimize
    res = minimize(cost,
                   init_params,
                   bounds=((-np.pi, np.pi), (-np.pi, np.pi), (-np.pi, np.pi), (-np.pi, np.pi)),
                   tol=1e-12)
    #print(res.x)
    params = res.x

    # set the initial parameter values
    # params = init_params
    # losses = [cost(params)]
    #
    # for i in range(steps):
    #     # update the circuit parameters
    #     # QHACK #
    #     params, prev_loss = opt.step_and_cost(cost, params)
    #     losses.append(cost(params))
    #
    #     if DEBUG and i % 20 == 0:
    #         print('Iteration = {:},  Energy = {:.8f} Ha'.format(i, losses[-1]))
    #     # QHACK #

    return winning_prob(params, alpha, beta)


if __name__ == '__main__':
    inputs = sys.stdin.read().split(",")
    output = optimize(float(inputs[0]), float(inputs[1]))
    print(f"{output}")