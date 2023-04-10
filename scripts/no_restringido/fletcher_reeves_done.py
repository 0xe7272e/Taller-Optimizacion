"""Método de Fletcher y Reeves (o método del gradiente conjugado)"""
import numpy as np



# Esta parte del backtracking para hayar el delta no la tengo muy clara
def backtracking_line_search(f, grad_f, x, d, alpha=0.001, rho=0.5, c=1e-4):
    '''
    Backtracking line search for finding a suitable step size.

    Inputs:
    f - objective function
    grad_f - gradient of objective function
    x - current point
    d - search direction
    alpha - initial step size (default: 1)
    rho - reduction factor (default: 0.5)
    c - Armijo condition constant (default: 1e-4)

    Outputs:
    alpha - suitable step size
    '''

    # Evaluate objective function and gradient at current point
    f_x = f(x)
    grad_f_x = grad_f(x)

    # Iterate until Armijo condition is satisfied
    while f(x + alpha * d) > f_x + c * alpha * np.dot(grad_f_x, d):
        alpha *= rho

    return alpha

def fletcher_reeves(x0, f, grad_f, max_iter=1000, tol=1e-6):
    '''
    Fletcher-Reeves method for unconstrained optimization.

    Inputs:
    x0 - initial guess as a 2-element list or array
    f - objective function
    grad_f - gradient of objective function
    max_iter - maximum number of iterations (default: 1000)
    tol - tolerance for convergence (default: 1e-6)

    Outputs:
    x - optimal point as a 2-element list or array
    f_x - optimal function value
    iter_num - number of iterations taken
    '''

    # Initialize variables
    x = np.array(x0)
    grad_prev = grad_f(x)
    d0 = -grad_prev
    iter_num = 0

    # Iterate until convergence or max number of iterations
    while iter_num < max_iter:

        # Line search
        alpha = backtracking_line_search(f, grad_f, x, d0)
        #alpha = 10**-3

        # Update x and gradient
        x_prev = x
        x = x + alpha * d0
        grad = grad_f(x)
        # Check for convergence
        if np.linalg.norm(grad) < tol:
            break

        # Update conjugate direction
        beta = np.dot(grad, grad) / np.dot(grad_prev, grad_prev)
        d0 = -grad + beta * d0

        # Update variables
        grad_prev = grad
        iter_num += 1

    # Compute final function value
    f_x = f(x)

    return x, f_x, iter_num


def f1(x):
    return float(x[0]**2 + x[1]**2 +1)

def grad_f1(x):
    return np.array([ float(  2*x[0] ) ,float( 2*x[1] )])
    #return np.array([2*x[0],2*x[1]])


punto, imagen, iteraciones = fletcher_reeves( [-1234567.0,98765.0], f1, grad_f1)
print(punto, imagen, iteraciones)
