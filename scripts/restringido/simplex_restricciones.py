import numpy as np
from scipy.optimize import minimize

# Definir la función objetivo
def objective_function(x):
    return (x[0] + x[1]**3) + 3*(x[0] + x[1])**4

# Definir la restricción
def constraint(x):
    return np.array([-x[0] + x[1] - 1, x[0] + x[1] - 2])

# Definir el método de restricciones
def constrained_minimization():
    # Definir el punto inicial
    x0 = np.array([1, 1])
    # Definir los límites de las variables
    bounds = [(0, None), (0, None)]
    # Definir las restricciones
    cons = [{'type': 'ineq', 'fun': constraint}]
    # Ejecutar la minimización con restricciones
    sol = minimize(objective_function, x0, method='SLSQP', bounds=bounds, constraints=cons)
    # Devolver la solución encontrada
    return sol.x

# Ejecutar el método de restricciones
sol = constrained_minimization()

# Imprimir la solución encontrada
print("Solución encontrada:")
print("x = ", sol[0])
print("y = ", sol[1])
