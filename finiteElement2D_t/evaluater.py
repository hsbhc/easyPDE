#-- coding: utf-8 --
'''
@Project: easyPDE
@File: evaluater.py
@Author: SryM.HJ
@Ide: PyCharm
@Time: 2021/12/6 21:00
@Function: Calculate the error  : maximum_error, Loo_norm_error, L2_norm_error, H1_semi_norm_error
'''

import numpy as np
import math
from finiteElement2D.solvers import Solver

class Evaluate():
    def __init__(self,solver: Solver):
        self.solver=solver
        self.matrixEquation=solver.matrixEquation
        self.PT_matrix=solver.PT_matrix
        self.linearSystem=solver.linearSystem
        self.question=solver.question

    @staticmethod
    def get_convergence_order(errors,hs=0.5):
        convergence_orders=[]
        for k in range(len(errors)):
            difference=0.0
            for i in range(len(errors[k])-1):
                difference+=(errors[k][i+1]/errors[k][i])
            convergence_order=math.log(difference/(len(errors[k])-1),hs)
            convergence_orders.append(round(convergence_order))
        return convergence_orders

    def get_maximum_error(self):
        if self.PT_matrix.En.dim==1:
            max_error = 0
            for i in range(self.PT_matrix.Nbm):
                temp = self.matrixEquation.solution[i] - self.question.U(self.PT_matrix.Pb[i])
                if abs(max_error) < abs(temp):
                    max_error = abs(temp)
            return max_error

        if self.PT_matrix.En.dim == 2:
            max_error = 0
            for i in range(self.PT_matrix.Nbm):
                x, y = self.PT_matrix.Pb[:, i]
                temp = self.matrixEquation.solution[i] - self.question.U(x,y)
                if abs(max_error) < abs(temp):
                    max_error = abs(temp)
            return max_error

    def Loo_norm_error(self):
        if self.PT_matrix.En.dim==1:
            error = 0
            for i in range(self.PT_matrix.N):
                En=self.PT_matrix.En.getEnByIndex(i)
                a = En.correct_En[0][0]
                b = En.correct_En[1][0]
                gauss_point = self.linearSystem.integrator.getX(En)
                analyticalSolution = self.question.U(gauss_point)
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
        if self.PT_matrix.En.dim ==2:
            error = 0
            for i in range(self.PT_matrix.N):
                En = self.PT_matrix.En.getEnByIndex(i)
                x,y = self.linearSystem.integrator.getX(En)
                analyticalSolution = self.question.U(x,y)
                numericalSolution = analyticalSolution.copy()
                for j in range(len(x)):
                    x_ = x[j]
                    y_ = y[j]
                    numericalSolution[j] = 0
                    for k in range(self.solver.Nlb_trial):
                        numericalSolution[j] += self.matrixEquation.solution[
                                                    self.PT_matrix.Tb[k, i]] * self.solver.trial(x_,y_,
                                                                                                 En.correct_En,
                                                                                                 k,
                                                                                                 0,0)
                error_i = np.max(np.abs(analyticalSolution - numericalSolution))
                if error_i > error:
                    error = error_i
            return error
    def L2_norm_error(self):
        if self.PT_matrix.En.dim==1:
            error = 0
            for i in range(self.PT_matrix.N):
                En=self.PT_matrix.En.getEnByIndex(i)
                a, b = En.correct_En[0][0], En.correct_En[1][0]
                gauss_point = self.linearSystem.integrator.getX(En)
                analyticalSolution = self.question.U(gauss_point)
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

                error += self.linearSystem.integrator.integral(En, f)
            return np.sqrt(error)

        if self.PT_matrix.En.dim == 2:
            error = 0
            for i in range(self.PT_matrix.N):
                En = self.PT_matrix.En.getEnByIndex(i)
                x,y = self.linearSystem.integrator.getX(En)
                analyticalSolution = self.question.U(x,y)
                numericalSolution = analyticalSolution.copy()
                for j in range(len(x)):
                    x_ = x[j]
                    y_=y[j]
                    numericalSolution[j] = 0
                    for k in range(self.solver.Nlb_trial):
                        numericalSolution[j] += self.matrixEquation.solution[
                                                    self.PT_matrix.Tb[k, i]] * self.solver.trial(x_,y_,
                                                                                                 En.correct_En,
                                                                                                 k,
                                                                                                 0,0)

                integrandFunction = np.square(analyticalSolution - numericalSolution)
                def f(x1,y1):
                    for l in range(len(x)):
                        if x[l]==x1 and y[l]==y1:
                            return integrandFunction[l]
                error += self.linearSystem.integrator.integral(En, f)
            return np.sqrt(error)

    def H1_semi_norm_error(self):
        if self.PT_matrix.En.dim==1:
            error = 0
            for i in range(self.PT_matrix.N):
                En = self.PT_matrix.En.getEnByIndex(i)
                a, b = En.correct_En[0][0], En.correct_En[1][0]
                gauss_point = self.linearSystem.integrator.getX(En)
                analyticalSolution = self.question.dU(gauss_point)
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

                error += self.linearSystem.integrator.integral(En, f)
            return np.sqrt(error)

        if self.PT_matrix.En.dim == 2:
            error = 0
            for i in range(self.PT_matrix.N):
                En = self.PT_matrix.En.getEnByIndex(i)
                x, y = self.linearSystem.integrator.getX(En)
                analyticalSolution = self.question.dUdx(x, y)
                numericalSolution = analyticalSolution.copy()
                for j in range(len(x)):
                    x_ = x[j]
                    y_ = y[j]
                    numericalSolution[j] = 0
                    for k in range(self.solver.Nlb_trial):
                        numericalSolution[j] += self.matrixEquation.solution[
                                                    self.PT_matrix.Tb[k, i]] * self.solver.trial(x_, y_,
                                                                                                 En.correct_En,
                                                                                                 k,
                                                                                                 1, 0)
                integrandFunction = np.square(analyticalSolution - numericalSolution)

                def f(x1, y1):
                    for l in range(len(x)):
                        if x[l] == x1 and y[l] == y1:
                            return integrandFunction[l]

                error += self.linearSystem.integrator.integral(En, f)
            for i in range(self.PT_matrix.N):
                En = self.PT_matrix.En.getEnByIndex(i)
                x, y = self.linearSystem.integrator.getX(En)
                analyticalSolution = self.question.dUdy(x, y)
                numericalSolution = analyticalSolution.copy()
                for j in range(len(x)):
                    x_ = x[j]
                    y_ = y[j]
                    numericalSolution[j] = 0
                    for k in range(self.solver.Nlb_trial):
                        numericalSolution[j] += self.matrixEquation.solution[
                                                    self.PT_matrix.Tb[k, i]] * self.solver.trial(x_, y_,
                                                                                                 En.correct_En,
                                                                                                 k,
                                                                                                 0, 1)
                integrandFunction = np.square(analyticalSolution - numericalSolution)

                def f(x1, y1):
                    for l in range(len(x)):
                        if x[l] == x1 and y[l] == y1:
                            return integrandFunction[l]

                error += self.linearSystem.integrator.integral(En, f)
            return np.sqrt(error)