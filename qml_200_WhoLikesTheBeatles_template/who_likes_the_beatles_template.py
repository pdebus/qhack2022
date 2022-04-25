#! /usr/bin/python3

import sys
from pennylane import numpy as np
import pennylane as qml


def distance(A, B):
    """Function that returns the distance between two vectors.

    Args:
        - A (list[int]): person's information: [age, minutes spent watching TV].
        - B (list[int]): person's information: [age, minutes spent watching TV].

    Returns:
        - (float): distance between the two feature vectors.
    """

    # QHACK #
    #print("A", A)
    #print("B", B)

    f = np.concatenate([A, B], axis=0)
    #print(f)
    # The Swap test is a method that allows you to calculate |<A|B>|^2 , you could use it to help you.
    # The qml.AmplitudeEmbedding operator could help you too.
    # from pennylane.transforms import merge_amplitude_embedding
    #
    # dev = qml.device("default.qubit", wires=3, shots=1000)
    # @qml.qnode(dev)
    # @merge_amplitude_embedding
    # def circuit(x, y):
    #     qml.AmplitudeEmbedding(features=x, wires=1, normalize=True)
    #     qml.AmplitudeEmbedding(features=y, wires=2, normalize=True)
    #     qml.Hadamard(wires=0)
    #     qml.CSWAP(wires=[0, 1, 2])
    #     qml.Hadamard(wires=0)
    #
    #     return qml.sample(qml.PauliZ(0))#qml.probs(wires=0)#
    #
    # probs = np.mean(circuit(A, B))#circuit(A, B)[0]#
    # overlap = np.sqrt(1 - 2 * probs)
    # dist = np.sqrt(2 * (1 - overlap))
    # print(probs, overlap, dist)

    normA = np.linalg.norm(A)
    normB = np.linalg.norm(B)
    a = np.array(A) / normA
    b = np.array(B) / normB

    dev = qml.device("default.qubit", wires=1, shots=100000)

    def embed(x, wires):
        qml.RY(2 * np.arccos(x[0]), wires=wires)

    @qml.qnode(dev)
    def circuit(x, y):
        embed(x, wires=0)
        qml.adjoint(embed)(y, wires=0)
        #qml.RY(2 * np.arccos(x[0]), wires=0)
        #qml.RY(-2 * np.arccos(y[0]), wires=0)
        return qml.probs()

    probs = np.mean([circuit(a, b)[0] for _ in range(1000)])
    overlap = np.sqrt(probs)
    dist = np.sqrt(2 * (1 - overlap))
    #print(probs, overlap, dist)

    return dist
    # QHACK #


def predict(dataset, new, k):
    """Function that given a dataset, determines if a new person do like Beatles or not.

    Args:
        - dataset (list): List with the age, minutes that different people watch TV, and if they like Beatles.
        - new (list(int)): Age and TV minutes of the person we want to classify.
        - k (int): number of nearby neighbors to be taken into account.

    Returns:
        - (str): "YES" if they like Beatles, "NO" otherwise.
    """

    # DO NOT MODIFY anything in this code block

    def k_nearest_classes():
        """Function that returns a list of k near neighbors."""
        distances = []
        for data in dataset:
            distances.append(distance(data[0], new))
        nearest = []
        for _ in range(k):
            indx = np.argmin(distances)
            nearest.append(indx)
            distances[indx] += 2

        return [dataset[i][1] for i in nearest]

    output = k_nearest_classes()

    return (
        "YES" if len([i for i in output if i == "YES"]) > len(output) / 2 else "NO",
        float(distance(dataset[0][0], new)),
    )


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block
    inputs = sys.stdin.read().split(",")
    dataset = []
    new = [int(inputs[0]), int(inputs[1])]
    k = int(inputs[2])
    for i in range(3, len(inputs), 3):
        dataset.append([[int(inputs[i + 0]), int(inputs[i + 1])], str(inputs[i + 2])])

    output = predict(dataset, new, k)
    sol = 0 if output[0] == "YES" else 1
    print(f"{sol},{output[1]}")
