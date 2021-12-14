import numpy as np
from Solver.question import U, dU
from Solver.solvers import Solver

class Evaluate():

    def __init__(self,solver: Solver):
        self.solver=solver
        self.matrixEquation=solver.matrixEquation
        self.PT_matrix=solver.PT_matrix
        self.linearSystem=solver.linearSystem

    def get_maximum_error(self):
        max_error = 0
        for i in range(self.PT_matrix.Nbm):
            temp = self.matrixEquation.solution[i] - U(self.PT_matrix.Pb[i])
            if abs(max_error) < abs(temp):
                max_error = abs(temp)
        return max_error

    def Loo_norm(self):
        error = 0
        for i in range(self.PT_matrix.N):
            a, b = self.PT_matrix.P[i], self.PT_matrix.P[i + 1]
            gauss_point = self.linearSystem.integrator.getX(a, b)
            analyticalSolution = U(gauss_point)
            numericalSolution = analyticalSolution.copy()
            for j in range(len(gauss_point)):
                x = gauss_point[j]
                numericalSolution[j] = 0
                for k in range(self.solver.Nlb_trial):
                    numericalSolution[j] += self.matrixEquation.solution[self.PT_matrix.Tb[k, i]] * self.solver.trial(x,
                                                                                                                     (a,
                                                                                                                      b),
                                                                                                                     k,
                                                                                                                     derivative=0)
            error_i = np.max(np.abs(analyticalSolution - numericalSolution))
            if error_i > error:
                error = error_i
        return error

    def L2_norm(self):
        error = 0
        for i in range(self.PT_matrix.N):
            a, b = self.PT_matrix.P[i], self.PT_matrix.P[i + 1]
            gauss_point = self.linearSystem.integrator.getX(a, b)
            analyticalSolution = U(gauss_point)
            numericalSolution = analyticalSolution.copy()
            for j in range(len(gauss_point)):
                x = gauss_point[j]
                numericalSolution[j] = 0
                for k in range(self.solver.Nlb_trial):
                    numericalSolution[j] += self.matrixEquation.solution[self.PT_matrix.Tb[k, i]] * self.solver.trial(x,
                                                                                                                     (a,
                                                                                                                      b),
                                                                                                                     k,
                                                                                                                     derivative=0)

            integrandFunction = np.square(analyticalSolution - numericalSolution)

            def f(x):
                index = np.where(gauss_point == x)
                return integrandFunction[index]

            error += self.linearSystem.integrator.integral(a, b, f)
        return np.sqrt(error)

    def H1_semi_norm(self):
        error = 0
        for i in range(self.PT_matrix.N):
            a, b = self.PT_matrix.P[i], self.PT_matrix.P[i + 1]
            gauss_point = self.linearSystem.integrator.getX(a, b)
            analyticalSolution = dU(gauss_point)
            numericalSolution = analyticalSolution.copy()
            for j in range(len(gauss_point)):
                x = gauss_point[j]
                numericalSolution[j] = 0
                for k in range(self.solver.Nlb_trial):
                    numericalSolution[j] += self.matrixEquation.solution[self.PT_matrix.Tb[k, i]] * self.solver.trial(x,
                                                                                                                     (a,
                                                                                                                      b),
                                                                                                                     k,
                                                                                                                     derivative=1)

            integrandFunction = np.square(analyticalSolution - numericalSolution)

            def f(x):
                index = np.where(gauss_point == x)
                return integrandFunction[index]

            error += self.linearSystem.integrator.integral(a, b, f)
        return np.sqrt(error)

