# Implementación de método simplex por Rodrigo Castillo Camargo.
import math
import numpy as np

# Definición de método para crear el tableau
def to_tableau(c, A, b):
    xb = [eq + [x] for eq, x in zip(A, b)]
    z = c + [0]
    return xb + [z]

# Regla para evaluar si el método puede ser mejorado
def can_be_improved(tableau):
    z = tableau[-1]
    return any(x > 0 for x in z[:-1])

# Método para encontrar el pivote en la cual iterará la siguiente iteracion
def get_pivot_position(tableau):
    z = tableau[-1]
    column = next(i for i, x in enumerate(z[:-1]) if x > 0)

    restrictions = []
    for eq in tableau[:-1]:
        el = eq[column]
        restrictions.append(math.inf if el <= 0 else eq[-1] / el)

    row = restrictions.index(min(restrictions))
    return row, column

# Método para modificar el Tableau
def pivot_step(tableau, pivot_position):
    new_tableau = [[] for eq in tableau]

    i, j = pivot_position
    pivot_value = tableau[i][j]
    new_tableau[i] = np.array(tableau[i]) / pivot_value

    for eq_i, eq in enumerate(tableau):
        if eq_i != i:
            multiplier = np.array(new_tableau[i]) * tableau[eq_i][j]
            new_tableau[eq_i] = np.array(tableau[eq_i]) - multiplier

    return new_tableau

# Método para mirar si la solución es básica
def is_basic(column):
    return sum(column) == 1 and len([c for c in column if c == 0]) == len(column) - 1

# Método para retornar la solución
def get_solution(tableau):
    columns = np.array(tableau).T
    solutions = []
    for column in columns[:-1]:
        solution = 0
        if is_basic(column):
            one_index = column.tolist().index(1)
            solution = columns[-1][one_index]
        solutions.append(solution)

    return solutions

# Definición del método simplex
def simplex(c, A, b):
    tableau = to_tableau(c, A, b)
    while can_be_improved(tableau):
        pivot_position = get_pivot_position(tableau)
        tableau = pivot_step(tableau, pivot_position)
    return get_solution(tableau)



if __name__ == "__main__":
    c = [1, 1, 0, 0, 0] # Función Objetivo
    A  = [
        [-1, 1, 1, 0, 0],
        [ 1, 0, 0, 1, 0],
        [ 0, 1, 0, 0, 1]
    ]
    b = [2, 4, 4] # Boundaries (límites de las restricciones)
    sol = simplex(c,A,b)
    print(f' la solución es {sol}')
