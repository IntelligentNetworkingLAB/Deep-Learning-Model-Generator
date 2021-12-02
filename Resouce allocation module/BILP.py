from gekko import GEKKO
import random


FinalResults = []
for iter in range(1, 5):
    m = GEKKO()
    m.options.SOLVER = 1
    m.solver_options = ['minlp_max_iter_with_int_sol 1000000','minlp_maximum_iterations 100000']
    
    '''Initail Coefficient'''
    C = [[random.uniform(1, 5) for col in range(iter)] for row in range(100)]
    print(C)


    '''Define Variable'''
    X = m.Array(m.Var, 100 * iter, integer=True)
    R = m.Array(m.Var, iter)
    Z = m.Var()

    '''Objective Function'''
    m.Maximize(Z)

    '''Constraints'''
    for i in X :
        i.lower = 0
    for i in range(int(len(X)/len(R))) :
        eq = 0
        for j in range(len(R)) :
            eq += X[i*len(R) + j]
        m.Equation(eq == 1)
    for i in range(len(R)) :
        eq = 0
        for j in range(int(len(X)/len(R))) :
            eq += C[j][i]*X[i + j*len(R)]
        m.Equation(eq == R[i])

    
    for i in R :
        m.Equation(Z<=i)

    '''Solve the problem'''
    m.solve()

    '''Print Results'''
    print('Indicator:')
    for i in range(int(len(X)/len(R))) :
        for j in range(len(R)) :
            print(X[i*len(R) + j].value[0],end=" ")
        print()
    print('Results:')
    for i in R :
        print(i, ': ', i.value[0])
    print('Min:   ',Z.value[0])
    FinalResults.append(Z.value[0])

print(FinalResults)