from scipy.optimize import minimize

def objective(x):
    return x[0]**2 + 3*x[1]**2 + 4*x[0]*x[1] + 6*x[0] + 8*x[1] + 9

def constraint1(x):
    return x[0] - x[1] - 1

def constraint2(x):
    return 2 - x[0] - x[1]

x0 = [0, 0]
bounds = ((0, None), (0, None))

cons = [{'type':'eq', 'fun':constraint1}, {'type':'ineq', 'fun':constraint2}]

result = minimize(objective, x0, method='BFGS', bounds=bounds, constraints=cons)

print(result)
