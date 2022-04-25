import sys
import pennylane as qml
from pennylane import numpy as np
import pennylane.optimize as optimize

DATA_SIZE = 250


def square_loss(labels, predictions):
    """Computes the standard square loss between model predictions and true labels.

    Args:
        - labels (list(int)): True labels (1/-1 for the ordered/disordered phases)
        - predictions (list(int)): Model predictions (1/-1 for the ordered/disordered phases)

    Returns:
        - loss (float): the square loss
    """

    loss = 0
    for l, p in zip(labels, predictions):
        loss = loss + (l - p) ** 2

    loss = loss / len(labels)
    return loss


def accuracy(labels, predictions):
    """Computes the accuracy of the model's predictions against the true labels.

    Args:
        - labels (list(int)): True labels (1/-1 for the ordered/disordered phases)
        - predictions (list(int)): Model predictions (1/-1 for the ordered/disordered phases)

    Returns:
        - acc (float): The accuracy.
    """

    acc = 0
    for l, p in zip(labels, predictions):
        if abs(l - p) < 1e-5:
            acc = acc + 1
    acc = acc / len(labels)

    return acc


def classify_ising_data(ising_configs, Y):
    """Learn the phases of the classical Ising model.

    Args:
        - ising_configs (np.ndarray): 250 rows of binary (0 and 1) Ising model configurations
        - Y (np.ndarray): 250 rows of labels (1 or -1)

    Returns:
        - predictions (list(int)): Your final model predictions

    Feel free to add any other functions than `cost` and `circuit` within the "# QHACK #" markers 
    that you might need.
    """

    # QHACK #

    num_wires = ising_configs.shape[1] 
    dev = qml.device("default.qubit", wires=num_wires, shots=1)

    # Define a variational circuit below with your needed arguments and return something meaningful
    @qml.qnode(dev)
    def circuit(params, x):

        num_spins = len(x)

        for n, s in enumerate(x):
            if s==1:
                qml.PauliX(n)

        # qml.templates.StronglyEntanglingLayers(params, wires=range(num_wires))
        return qml.sample(wires=range(num_spins))
        #return [qml.expval(qml.PauliZ(i)) for i in range(num_spins)]
        #measurements = np.array()
        #M = np.sum(2 * measurements - 1.0) / num_spins

        #return M

    # Define a cost function below with your needed arguments
    def cost(params):

        # QHACK #
        
        # Insert an expression for your model predictions here
        predictions = []

        for x in ising_configs:
            M = circuit(params, x)
            y = 1 if abs(M) > 0.5 else -1
            predictions.append(y)

        # QHACK #

        return square_loss(Y, predictions) # DO NOT MODIFY this line

    # optimize your circuit here
    params = None
    predictions = []

    for x in ising_configs:
        measurements = np.array(circuit(params, x))
        M = np.sum(2 * measurements - 1.0) / num_wires
        #print(measurements, M)
        y = 1 if abs(M) > 0.5 else -1
        predictions.append(y)

    #print(accuracy(Y, predictions))

    # QHACK #

    return predictions


if __name__ == "__main__":
    inputs = np.array(
        sys.stdin.read().split(","), dtype=int, requires_grad=False
    ).reshape(DATA_SIZE, -1)
    ising_configs = inputs[:, :-1]
    labels = inputs[:, -1]
    predictions = classify_ising_data(ising_configs, labels)
    print(*predictions, sep=",")
