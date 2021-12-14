#-- coding: utf-8 --
'''
@Project: easyPDE
@File: main.py
@Author: SryM.HJ
@Ide: PyCharm
@Time: 2021/12/6 21:04
@Function: test finiteElement1D
'''
import numpy as np
from finiteElement1D.PTmatrix import PMatrixType, TMatrixType
from finiteElement1D.basisFunction import BasisFunctionType
from finiteElement1D.evaluater import Evaluate
from finiteElement1D.integrators import IntegratorType
from finiteElement1D.question import Question
from finiteElement1D.solvers import Solver


class Question_Dirichlet(Question):
    def G(self, x):
        '''
        Dirichlet boundary condition
        U(a) = G(a)
        U(b) = G(b)
        '''
        if x == 0:
            return 0
        if x == 1:
            return np.cos(1)
        return 'No limit'


class Question_Neumann(Question):
    def G(self, x):
        '''
        Neumann boundary condition
        U(a) = G(a)
        '''
        if x == 0:
            return 0
        else:
            return 'No limit'

    def dG(self, x):
        '''
        Neumann boundary condition
        dU(b)/dx = G(b)
        '''
        if x == 1:
            return np.cos(1) - np.sin(1)
        else:
            return 'No limit'


class Question_Robin(Question):
    def G(self, x):
        '''
        Robin boundary condition
        U(b) = G(b)
        '''
        if x == 1:
            return np.cos(1)
        else:
            return 'No limit'

    def dGQbPb(self, x):
        '''
        Robin boundary condition
        dU(a)/dx + Q*U(b) = P*U(b)
        '''
        Q = 1
        P = 1
        if x == 0:
            return Q, P
        else:
            return 'No limit'


def example1():
    print('     %40s'%('example1--Dirichlet--One_Dimensional_Linear_Element'))
    print('-'*62)
    print('%2s     %11s      %10s   %10s    %10s' % ('h', 'maxerror', '||U-Uh||oo', '||U-Uh||0', '||U-Uh||1'))
    errors=[[],[],[],[]]
    for i in range(2, 8):
        h = 1 / (2 ** i)
        question = Question_Dirichlet()

        solver = Solver(question=question,h_Mesh=h,h_Finite=h)
        solver.setBasisFunction(trialBasisFunctionType=BasisFunctionType.One_Dimensional_Linear_Element,
                                    testBasisFunctionType=BasisFunctionType.One_Dimensional_Linear_Element)
        solver.setIntegrator(IntegratorType.Gauss)
        solver.integrator.set_gauss_point_number(4)
        solver.setPT_PbTb(PMatrixType.One_Dimensional_Linear_Node, TMatrixType.One_Dimensional_Linear_Cell,
                      PMatrixType.One_Dimensional_Linear_Node, TMatrixType.One_Dimensional_Linear_Cell)

        solver.makeLinearSystem()
        solver.makeBoundaryProcess()
        solver.solve()

        evaluate = Evaluate(solver)
        print('1/%-3s    %.4e     %.4e    %.4e    %.4e' % (
        str(2 ** i), evaluate.get_maximum_error(), evaluate.Loo_norm_error(), evaluate.L2_norm_error(),
        evaluate.H1_semi_norm_error()))

        errors[0].append(evaluate.get_maximum_error())
        errors[1].append(evaluate.Loo_norm_error())
        errors[2].append(evaluate.L2_norm_error())
        errors[3].append(evaluate.H1_semi_norm_error())
    orders=Evaluate.get_convergence_order(errors)
    print(' O         O(h^%d)         O(h^%d)        O(h^%d)        O(h^%d)' % (orders[0],orders[1],orders[2],orders[3]))
    print('-' * 62)
def example2():
    print('     %40s' % ('example2--Dirichlet--One_Dimensional_Quadratic_Element'))
    print('-' * 62)
    print('%2s     %11s      %10s   %10s    %10s' % ('h', 'maxerror', '||U-Uh||oo', '||U-Uh||0', '||U-Uh||1'))
    errors = [[], [], [], []]
    for i in range(2, 8):
        h = 1 / (2 ** i)
        question = Question_Dirichlet()

        solver = Solver(question=question,h_Mesh=h,h_Finite=h/2)
        solver.setBasisFunction(trialBasisFunctionType=BasisFunctionType.One_Dimensional_Quadratic_Element,
                                    testBasisFunctionType=BasisFunctionType.One_Dimensional_Quadratic_Element)
        solver.setIntegrator(IntegratorType.Gauss)
        solver.integrator.set_gauss_point_number(4)
        solver.setPT_PbTb(PMatrixType.One_Dimensional_Linear_Node, TMatrixType.One_Dimensional_Linear_Cell,
                      PMatrixType.One_Dimensional_Linear_Node, TMatrixType.One_Dimensional_Quadratic_Cell)

        solver.makeLinearSystem()
        solver.makeBoundaryProcess()
        solver.solve()

        evaluate = Evaluate(solver)
        print('1/%-3s    %.4e     %.4e    %.4e    %.4e' % (str(2 ** i), evaluate.get_maximum_error(), evaluate.Loo_norm_error(), evaluate.L2_norm_error(),evaluate.H1_semi_norm_error()))
        errors[0].append(evaluate.get_maximum_error())
        errors[1].append(evaluate.Loo_norm_error())
        errors[2].append(evaluate.L2_norm_error())
        errors[3].append(evaluate.H1_semi_norm_error())
    orders=Evaluate.get_convergence_order(errors)
    print(' O         O(h^%d)         O(h^%d)        O(h^%d)        O(h^%d)' % (orders[0],orders[1],orders[2],orders[3]))
    print('-' * 62)
def example3():
    print('     %40s' % ('example3--Neumann--One_Dimensional_Quadratic_Element'))
    print('-' * 62)
    print('%2s     %11s      %10s   %10s    %10s' % ('h', 'maxerror', '||U-Uh||oo', '||U-Uh||0', '||U-Uh||1'))
    errors = [[], [], [], []]
    for i in range(2, 8):
        h = 1 / (2 ** i)
        question = Question_Neumann()

        solver = Solver(question=question,h_Mesh=h,h_Finite=h/2)
        solver.setBasisFunction(trialBasisFunctionType=BasisFunctionType.One_Dimensional_Quadratic_Element,
                                    testBasisFunctionType=BasisFunctionType.One_Dimensional_Quadratic_Element)
        solver.setIntegrator(IntegratorType.Gauss)
        solver.integrator.set_gauss_point_number(4)
        solver.setPT_PbTb(PMatrixType.One_Dimensional_Linear_Node, TMatrixType.One_Dimensional_Linear_Cell,
                      PMatrixType.One_Dimensional_Linear_Node, TMatrixType.One_Dimensional_Quadratic_Cell)

        solver.makeLinearSystem()
        solver.makeBoundaryProcess()
        solver.solve()

        evaluate = Evaluate(solver)
        print('1/%-3s    %.4e     %.4e    %.4e    %.4e' % (
            str(2 ** i), evaluate.get_maximum_error(), evaluate.Loo_norm_error(), evaluate.L2_norm_error(),
            evaluate.H1_semi_norm_error()))
        errors[0].append(evaluate.get_maximum_error())
        errors[1].append(evaluate.Loo_norm_error())
        errors[2].append(evaluate.L2_norm_error())
        errors[3].append(evaluate.H1_semi_norm_error())
    orders = Evaluate.get_convergence_order(errors)
    print(' O         O(h^%d)         O(h^%d)        O(h^%d)        O(h^%d)' % (orders[0], orders[1], orders[2], orders[3]))
    print('-' * 62)
def example4():
    print('     %40s' % ('example4--Robin--One_Dimensional_Quadratic_Element'))
    print('-' * 62)
    print('%2s     %11s      %10s   %10s    %10s' % ('h', 'maxerror', '||U-Uh||oo', '||U-Uh||0', '||U-Uh||1'))
    errors = [[], [], [], []]
    for i in range(1, 5):
        h = 1 / (4 ** i)
        question = Question_Robin()

        solver = Solver(question=question,h_Mesh=h,h_Finite=h/2)
        solver.setBasisFunction(trialBasisFunctionType=BasisFunctionType.One_Dimensional_Quadratic_Element,
                                    testBasisFunctionType=BasisFunctionType.One_Dimensional_Quadratic_Element)
        solver.setIntegrator(IntegratorType.Gauss)
        solver.integrator.set_gauss_point_number(4)
        solver.setPT_PbTb(PMatrixType.One_Dimensional_Linear_Node, TMatrixType.One_Dimensional_Linear_Cell,
                      PMatrixType.One_Dimensional_Linear_Node, TMatrixType.One_Dimensional_Quadratic_Cell)

        solver.makeLinearSystem()
        solver.makeBoundaryProcess()
        solver.solve()

        evaluate = Evaluate(solver)
        print('1/%-3s    %.4e     %.4e    %.4e    %.4e' % (
            str(4 ** i), evaluate.get_maximum_error(), evaluate.Loo_norm_error(), evaluate.L2_norm_error(),
            evaluate.H1_semi_norm_error()))
        errors[0].append(evaluate.get_maximum_error())
        errors[1].append(evaluate.Loo_norm_error())
        errors[2].append(evaluate.L2_norm_error())
        errors[3].append(evaluate.H1_semi_norm_error())
    orders = Evaluate.get_convergence_order(errors,hs=0.25)
    print(' O         O(h^%d)         O(h^%d)        O(h^%d)        O(h^%d)' % (orders[0], orders[1], orders[2], orders[3]))
    print('-' * 62)


if __name__ == '__main__':
    example1()
    example2()
    example3()
    example4()



