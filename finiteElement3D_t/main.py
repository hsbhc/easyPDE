# -- coding: utf-8 --
'''
@Project: easyPDE
@File: main.py
@Author: SryM.HJ
@Ide: PyCharm
@Time: 2021/12/6 21:04
@Function: test finiteElement
'''
import numpy as np
from finiteElement3D_t.PTmatrix import PMatrixType, TMatrixType
from finiteElement3D_t.basisFunction import BasisFunctionType
from finiteElement3D_t.evaluater import Evaluate_t
from finiteElement3D_t.integrators import IntegratorType
from finiteElement3D_t.solvers import Solver_t
from finiteElement2D_t.question import One_Dimensional_PoissonQuestion, Two_Dimensional_PoissonQuestion, \
    Two_Dimensional_PoissonQuestion_t
import time

from finiteElement3D_t.question import Three_Dimensional_PoissonQuestion_t

np.set_printoptions(linewidth=np.inf)


class Question_Dirichlet_3D_t(Three_Dimensional_PoissonQuestion_t):
    def __init__(self, range=((0, 1),(0,1),(0,1)),t_range=(0,1)):
        super().__init__()
        self.range=range
        self.t_range=t_range

    def C(self, x, y,z,t):
        return 1

    def F(self, x, y,z,t):
        return -2 * np.exp(x + y+z+t)

    '''
    Analytical solution
    '''

    def U(self, x, y,z,t):
        return np.exp(x + y+z+t)

    def dUdx(self, x, y,z,t):
        return np.exp(x + y+z+t)

    def dUdy(self, x, y,z,t):
        return np.exp(x + y+z+t)

    def dUdt(self, x, y,z,t):
        return np.exp(x + y+z+t)
    def dUdz(self, x, y,z,t):
        return np.exp(x + y+z+t)

    def G(self, x, y,z,t):
        '''
        Dirichlet boundary condition
        '''
        if x == 0:
            return np.exp(t + y+z)
        if x == 1:
            return np.exp(1 + y+z+t)
        if y == 0:
            return np.exp(t + x+z)
        if y == 1:
            return np.exp(1+t + x+z)
        if z == 0:
            return np.exp(t + x+y)
        if z == 1:
            return np.exp(1+t + x+y)
        return 'No limit'

    def T0(self, x, y,z):
        return np.exp(x+y+z)





def example1():
    print('       %40s' % ('scheme=1/2--D--Two_Dimensional_Linear_Element --t'))
    print('-' * 80)
    print('%2s     %2s    %11s      %10s   %10s    %10s      time' % (
    'h', 't', 'maxerror', '||U-Uh||oo', '||U-Uh||0', '||U-Uh||1'))
    errors = [[], [], [], []]
    for i in range(2, 5):
        start = time.time()
        h = 1 / (2 ** i)
        question = Question_Dirichlet_3D_t()
        solver = Solver_t(question=question, h_Mesh=(h, h,h), h_Finite=(h, h,h), h_t=h, scheme=1 / 2)
        solver.setBasisFunction(trialBasisFunctionType=BasisFunctionType.Three_Dimensional_Linear_Element,
                                testBasisFunctionType=BasisFunctionType.Three_Dimensional_Linear_Element)
        solver.setIntegrator(IntegratorType.Gauss)
        solver.integrator.set_gauss_point_number(8)
        solver.setPT_PbTb(PMatrixType.Three_Dimensional_Linear_Node, TMatrixType.Three_Dimensional_Linear_Cell,
                          PMatrixType.Three_Dimensional_Linear_Node, TMatrixType.Three_Dimensional_Linear_Cell)

        solver.makeLinearSystem()

        end = time.time()
        evaluate = Evaluate_t(solver)
        print('1/%-3s  1/%-3s   %.4e     %.4e    %.4e    %.4e    %.4f' % (
            str(2 ** i), str(2 ** i), evaluate.get_maximum_error(), evaluate.Loo_norm_error(),
            evaluate.L2_norm_error(),
            evaluate.H1_semi_norm_error(), end - start))
        errors[0].append(evaluate.get_maximum_error())
        errors[1].append(evaluate.Loo_norm_error())
        errors[2].append(evaluate.L2_norm_error())
        errors[3].append(evaluate.H1_semi_norm_error())
    start = time.time()
    orders = Evaluate_t.get_convergence_order(errors)
    end = time.time()
    print(' O               O(h^%d)         O(h^%d)        O(h^%d)        O(h^%d)      %.4f' % (
        orders[0], orders[1], orders[2], orders[3], end - start))
    print('-' * 80)

def example2():
    print('       %40s' % ('scheme=1--D--Two_Dimensional_linear_Element --t'))
    print('-' * 80)
    print('%2s     %2s    %11s      %10s   %10s    %10s      time' % (
    'h', 't', 'maxerror', '||U-Uh||oo', '||U-Uh||0', '||U-Uh||1'))
    errors = [[], [], [], []]
    for i in range(2, 5):
        start = time.time()
        h = 1 / (2 ** i)
        question = Question_Dirichlet_3D_t()
        solver = Solver_t(question=question, h_Mesh=(h, h,h), h_Finite=(h, h,h), h_t=4*(h**2), scheme=1)
        solver.setBasisFunction(trialBasisFunctionType=BasisFunctionType.Three_Dimensional_Linear_Element,
                                testBasisFunctionType=BasisFunctionType.Three_Dimensional_Linear_Element)
        solver.setIntegrator(IntegratorType.Gauss)
        solver.integrator.set_gauss_point_number(8)
        solver.setPT_PbTb(PMatrixType.Three_Dimensional_Linear_Node, TMatrixType.Three_Dimensional_Linear_Cell,
                          PMatrixType.Three_Dimensional_Linear_Node, TMatrixType.Three_Dimensional_Linear_Cell)

        solver.makeLinearSystem()

        end = time.time()
        evaluate = Evaluate_t(solver)
        print('1/%-3s  1/%-3s   %.4e     %.4e    %.4e    %.4e    %.4f' % (
            str(2 ** i), str(4*((2 ** i)**2)), evaluate.get_maximum_error(), evaluate.Loo_norm_error(),
            evaluate.L2_norm_error(),
            evaluate.H1_semi_norm_error(), end - start))
        errors[0].append(evaluate.get_maximum_error())
        errors[1].append(evaluate.Loo_norm_error())
        errors[2].append(evaluate.L2_norm_error())
        errors[3].append(evaluate.H1_semi_norm_error())
    start = time.time()
    orders = Evaluate_t.get_convergence_order(errors)
    end = time.time()
    print(' O               O(h^%d)         O(h^%d)        O(h^%d)        O(h^%d)      %.4f' % (
        orders[0], orders[1], orders[2], orders[3], end - start))
    print('-' * 80)

if __name__ == '__main__':
    example1()
    example2()
