#! /usr/bin/python3

import sys
from pennylane import numpy as np
import pennylane as qml


def deutsch_jozsa(fs):
    """Function that determines whether four given functions are all of the same type or not.

    Args:
        - fs (list(function)): A list of 4 quantum functions. Each of them will accept a 'wires' parameter.
        The first two wires refer to the input and the third to the output of the function.

    Returns:
        - (str) : "4 same" or "2 and 2"
    """

    # QHACK #
    dev = qml.device("default.qubit", wires=8, shots=1)
    @qml.qnode(dev)
    def circuit():
        qml.PauliX(wires=2)
        for i in range(3):
            qml.Hadamard(wires=i)

        for n, f in enumerate(fs):
            f(wires=[0, 1, 2])
            for i in range(2):
                qml.Hadamard(wires=i)
                qml.PauliX(wires=i)

            qml.Toffoli(wires=[0, 1, 3 + n])
            # uncompute
            for i in range(2):
                qml.PauliX(wires=i)
                qml.Hadamard(wires=i)
            qml.adjoint(f)(wires=[0, 1, 2])

        return qml.sample(wires=[3, 4, 5, 6])

    measurement = circuit()

    #print(measurement)
    result = "4 same" if np.sum(measurement) == 0 or np.sum(measurement) == 4 else "2 and 2"

    return result
    # QHACK #


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block
    inputs = sys.stdin.read().split(",")
    numbers = [int(i) for i in inputs]

    # Definition of the four oracles we will work with.

    def f1(wires):
        qml.CNOT(wires=[wires[numbers[0]], wires[2]])
        qml.CNOT(wires=[wires[numbers[1]], wires[2]])

    def f2(wires):
        qml.CNOT(wires=[wires[numbers[2]], wires[2]])
        qml.CNOT(wires=[wires[numbers[3]], wires[2]])

    def f3(wires):
        qml.CNOT(wires=[wires[numbers[4]], wires[2]])
        qml.CNOT(wires=[wires[numbers[5]], wires[2]])
        qml.PauliX(wires=wires[2])

    def f4(wires):
        qml.CNOT(wires=[wires[numbers[6]], wires[2]])
        qml.CNOT(wires=[wires[numbers[7]], wires[2]])
        qml.PauliX(wires=wires[2])

    output = deutsch_jozsa([f1, f2, f3, f4])
    print(f"{output}")
