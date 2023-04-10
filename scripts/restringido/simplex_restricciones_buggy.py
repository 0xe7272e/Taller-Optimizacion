import numpy as np

# Definir la función objetivo
def objective_function(x, y):
    return (x + y**3) + 3*(x + y)**4

# Definir la restricción
def constraint(x, y):
    return np.array([-x + y - 1, x + y - 2])

# Definir el método del gradiente descendente con restricciones
def constrained_gradient_descent(x0, alpha, max_iter):
    # Inicializar la solución
    x = x0.copy()
    # Inicializar el contador de iteraciones
    it = 0
    # Repetir hasta que se alcance el número máximo de iteraciones
    while it < max_iter:
        # Calcular el gradiente de la función objetivo
        y = x[1]
        grad = np.array([1 + 12*(x[0] + y)**3 + 3*(x[0] + y)**4,
                         3*y**2 + 12*(x[0] + y)**3 + 3*(x[0] + y)**4])
        # Proyectar la solución en el conjunto factible de soluciones permitidas por las restricciones
        x = np.maximum(x - alpha * grad, 0)
        # Verificar si la solución cumple con las restricciones
        c = constraint(x[0], x[1])
        if np.all(c <= 0):
            return x
        else:
            # Calcular el paso de aprendizaje adaptativo
            alpha *= 0.5
        # Incrementar el contador de iteraciones
        it += 1
    # Si no se encuentra una solución que cumpla con las restricciones, devolver None
    return None

# Definir el punto inicial y la tasa de aprendizaje
x0 = np.array([1, 1])
alpha = 0.1
max_iter = 1000

# Ejecutar el método del gradiente descendente con restricciones
sol = constrained_gradient_descent(x0, alpha, max_iter)

# Imprimir la solución encontrada
if sol is not None:
    print("Solución encontrada:")
    print("x = ", sol[0])
    print("y = ", sol[1])
else:
    print("No se encontró una solución que cumpla con las restricciones.")
