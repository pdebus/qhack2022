#! /usr/bin/python3

import sys
import pennylane as qml
from pennylane import numpy as np

dev = qml.device("default.qubit", wires=1, shots=1)


@qml.qnode(dev)
def is_bomb(angle):
    """Construct a circuit at implements a one shot measurement at the bomb.

    Args:
        - angle (float): transmissivity of the Beam splitter, corresponding
        to a rotation around the Y axis.

    Returns:
        - (np.ndarray): a length-1 array representing result of the one-shot measurement
    """

    # QHACK #
    qml.RY(2 * angle, wires=0)
    # QHACK #

    return qml.sample(qml.PauliZ(0))


@qml.qnode(dev)
def bomb_tester(angle):
    """Construct a circuit that implements a final one-shot measurement, given that the bomb does not explode

    Args:
        - angle (float): transmissivity of the Beam splitter right before the final detectors

    Returns:
        - (np.ndarray): a length-1 array representing result of the one-shot measurement
    """

    # QHACK #
    qml.RY(2 * angle, wires=0)
    qml.RY(2 * angle, wires=0)
    # QHACK #

    return qml.sample(qml.PauliZ(0))


def simulate(angle, n):
    """Concatenate n bomb circuits and a final measurement, and return the results of 10000 one-shot measurements

    Args:
        - angle (float): transmissivity of all the beam splitters, taken to be identical.
        - n (int): number of bomb circuits concatenated

    Returns:
        - (float): number of bombs successfully tested / number of bombs that didn't explode.
    """

    # QHACK #
    # @qml.qnode(dev)
    # def concat_circuit():
    #     qml.RY(2 * angle, wires=0)
    #     for i in range(n):
    #         qml.RY(2 * angle, wires=0)
    #     return qml.sample(qml.PauliZ(0))
    #
    # if n == 1:
    #     shots = (-np.array(is_bomb(angle, shots=100000)) + 1) / 2
    # else:
    #     shots = (-np.array(concat_circuit(shots=100000)) + 1) / 2
    # p = np.mean(shots).round(2)
    num_one_shots = 10000

    explosions = 0
    no_explosions = 0
    detectorC = 0
    detectorD = 0

    # for i in range(num_one_shots):
    #     for j in range(1, n + 1):
    #         shot = is_bomb(angle)
    #         if shot == 1:
    #             explosions += 1
    #             break
    #     if shot == -1:
    #         detector_shot = bomb_tester(angle)
    #         no_explosions += 1
    #         if detector_shot == 1:
    #             detectorD += 1
    #         else:
    #             detectorC += 1

    from collections import defaultdict

    counts = defaultdict(int)
    for i in range(num_one_shots):
        measurement = "".join([str((is_bomb(angle)+1)//2) for j in range(n + 1)])
        counts[measurement] += 1

    detectorD = counts['0' * (n + 1)]
    detectorC = counts['0' * n + '1']
    no_explosions = detectorC + detectorD
    explosions = num_one_shots - no_explosions

    #print(counts)
    #print(explosions, no_explosions, detectorC, detectorD)

    p = detectorD / no_explosions

    return p
    # QHACK #


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block
    inputs = sys.stdin.read().split(",")
    output = simulate(float(inputs[0]), int(inputs[1]))
    print(f"{output}")
