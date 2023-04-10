import numpy as np
from scipy.optimize import line_search


# Davison Fletcher Powell
def dfp_optimizer(f, grad_f, x0, eps=1e-6, max_iter=1000):
    """
    DFP optimization algorithm for smooth convex functions.

    Parameters:
    f (callable): Objective function to be minimized.
    grad_f (callable): Gradient of the objective function.
    x0 (array-like): Initial guess for the minimum of the function.
    eps (float, optional): Tolerance for convergence. Defaults to 1e-6.
    max_iter (int, optional): Maximum number of iterations. Defaults to 1000.

    Returns:
    x (array-like): Estimated minimum of the function.
    f_x (float): Value of the function at the estimated minimum.
    iter (int): Number of iterations taken to converge.
    """
    # Initialization
    x = np.array(x0)
    H = np.eye(len(x))
    grad = grad_f(x)
    iter = 0

    # Main loop
    while np.linalg.norm(grad) > eps and iter < max_iter:
        p = -np.dot(H, grad)
        alpha = line_search(f, grad_f, x, p)[0]
        x_new = x + alpha * p
        grad_new = grad_f(x_new)
        s = x_new - x
        y = grad_new - grad
        rho = 1 / np.dot(y, s)
        H = np.dot((np.eye(len(x)) - rho * np.outer(s, y)), np.dot(H, (np.eye(len(x)) - rho * np.outer(y, s)))) + rho * np.outer(s, s)
        x, grad = x_new, grad_new
        iter += 1
    f_x = f(x)
    return x, f_x, iter


def f(x):
    return x[0]**2 + x[1]**2 + 5

def grad_f(x):
    return np.array([2*x[0], 2*x[1]])

x0 = [1, 1]
x, f_x, iter = dfp_optimizer(f, grad_f, x0)
print("Minimum value: ", f_x)
print("Estimated minimum: ", x)
print("Number of iterations: ", iter)
