#! /usr/bin/python3

import sys
import numpy as np
from scipy.optimize import minimize


def givens_rotations(a, b, c, d):
    """Calculates the angles needed for a Givens rotation to out put the state with amplitudes a,b,c and d

    Args:
        - a,b,c,d (float): real numbers which represent the amplitude of the relevant basis states (see problem statement). Assume they are normalized.

    Returns:
        - (list(float)): a list of real numbers ranging in the intervals provided in the challenge statement, which represent the angles in the Givens rotations,
        in order, that must be applied.
    """

    # QHACK #
    def check(t1, t2, t3):
        f1 = np.cos(0.5 * t1) * np.cos(0.5 * t3)
        f2 = - np.sin(0.5 * t1) * np.cos(0.5 * t2)
        f3 = np.sin(0.5 * t1) * np.sin(0.5 * t2)
        f4 = - np.cos(0.5 * t1) * np.sin(0.5 * t3)
        return [f1, f2, f3, f4]

    def func(t1, t2, t3):
        f1 = np.cos(0.5 * t1) * np.cos(0.5 * t3) - a
        f2 = - np.sin(0.5 * t1) * np.cos(0.5 * t2) - b
        f3 = np.sin(0.5 * t1) * np.sin(0.5 * t2) - c
        f4 = - np.cos(0.5 * t1) * np.sin(0.5 * t3) - d

        return f1 ** 2 + f2 ** 2 + f3 ** 2 + f4 ** 2

    func_val = 100
    max_iter = 500
    tol = 1e-10

    restarts = 0
    np.random.seed(42)
    while func_val > tol and restarts < max_iter:
        x0 = np.random.uniform(-np.pi, np.pi, 3)
        res = minimize(lambda x: func(x[0], x[1], x[2]),
                       x0,
                       bounds=((-np.pi, np.pi), (-np.pi, np.pi), (-np.pi, np.pi)),
                       tol=1e-12)
        func_val = func(*res.x)
        restarts += 1

    # print(res.x)
    # print(func_val)
    #
    # print(check(*res.x))
    # print(a, b, c, d)

    #print(check(1.0701416143903084,-0.39479111969976155,0.7124798514013161))
    return res.x
    # QHACK #


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block
    inputs = sys.stdin.read().split(",")
    theta_1, theta_2, theta_3 = givens_rotations(
        float(inputs[0]), float(inputs[1]), float(inputs[2]), float(inputs[3])
    )
    print(*[theta_1, theta_2, theta_3], sep=",")
