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
from finiteElement2D.PTmatrix import PMatrixType, TMatrixType
from finiteElement2D.basisFunction import BasisFunctionType
from finiteElement2D.evaluater import Evaluate
from finiteElement2D.integrators import IntegratorType
from finiteElement2D.solvers import Solver
from finiteElement2D.question import One_Dimensional_PoissonQuestion, Two_Dimensional_PoissonQuestion
import time

np.set_printoptions(linewidth=np.inf)


class Question_Dirichlet(One_Dimensional_PoissonQuestion):
    def __init__(self, range=(0, 1)):
        super().__init__(range)

    def C(self, x):
        return np.exp(x)

    def F(self, x):
        return -np.exp(x) * (np.cos(x) - 2 * np.sin(x) - x * np.cos(x) - x * np.sin(x))

    '''
    Analytical solution
    '''

    def U(self, x):
        return x * np.cos(x)

    def dU(self, x):
        return np.cos(x) - x * np.sin(x)

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


class Question_Neumann(One_Dimensional_PoissonQuestion):
    def __init__(self, range=(0, 1)):
        super().__init__(range)

    def C(self, x):
        return np.exp(x)

    def F(self, x):
        return -np.exp(x) * (np.cos(x) - 2 * np.sin(x) - x * np.cos(x) - x * np.sin(x))

    '''
    Analytical solution
    '''

    def U(self, x):
        return x * np.cos(x)

    def dU(self, x):
        return np.cos(x) - x * np.sin(x)

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


class Question_Robin(One_Dimensional_PoissonQuestion):
    def __init__(self, range=(0, 1)):
        super().__init__(range)

    def C(self, x):
        return np.exp(x)

    def F(self, x):
        return -np.exp(x) * (np.cos(x) - 2 * np.sin(x) - x * np.cos(x) - x * np.sin(x))

    '''
    Analytical solution
    '''

    def U(self, x):
        return x * np.cos(x)

    def dU(self, x):
        return np.cos(x) - x * np.sin(x)

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


class Question_Dirichlet_2D(Two_Dimensional_PoissonQuestion):
    def __init__(self, range=((-1, 1), (-1, 1))):
        super().__init__(range)

    def C(self, x, y):
        return 1

    def F(self, x, y):
        return -y * (1 - y) * (1 - x - (x ** 2) / 2) * np.exp(x + y) - x * (1 - x / 2) * (-3 * y - y ** 2) * np.exp(
            x + y)

    '''
    Analytical solution
    '''

    def U(self, x, y):
        return x * y * (1 - x / 2) * (1 - y) * np.exp(x + y)

    def dUdx(self, x, y):
        return (y * (1 - x / 2) * (1 - y) - 0.5 * x * y * (1 - y) + x * y * (1 - x / 2) * (1 - y)) * np.exp(x + y)

    def dUdy(self, x, y):
        return (x * (1 - x / 2) * (1 - y) - x * y * (1 - x / 2) + x * y * (1 - x / 2) * (1 - y)) * np.exp(x + y)

    def G(self, x, y):
        '''
        Dirichlet boundary condition
        '''
        if x == -1:
            return -1.5 * y * (1 - y) * np.exp(-1 + y)
        if x == 1:
            return 0.5 * y * (1 - y) * np.exp(1 + y)
        if y == -1:
            return -2 * x * (1 - x / 2) * np.exp(x - 1)
        if y == 1:
            return 0
        return 'No limit'


class Question_Neumann_2D(Two_Dimensional_PoissonQuestion):
    def __init__(self, range=((-1, 1), (-1, 1))):
        super().__init__(range)

    def C(self, x, y):
        return 1

    def F(self, x, y):
        return -2 * np.exp(x + y)

    '''
    Analytical solution
    '''

    def U(self, x, y):
        return np.exp(x + y)

    def dUdx(self, x, y):
        return np.exp(x + y)

    def dUdy(self, x, y):
        return np.exp(x + y)

    def G(self, x, y):
        '''
        Dirichlet boundary condition
        '''
        if x == -1:
            return np.exp(-1 + y)
        if x == 1:
            return np.exp(1 + y)
        if y == 1:
            return np.exp(1 + x)
        return 'No limit'

    def dG(self, x, y):
        if y == -1:
            return -1 * np.exp(x - 1)
        else:
            return 'No limit'


class Question_Robin_2D(Two_Dimensional_PoissonQuestion):
    def __init__(self, range=((-1, 1), (-1, 1))):
        super().__init__(range)

    def C(self, x, y):
        return 1

    def F(self, x, y):
        return -2 * np.exp(x + y)

    '''
    Analytical solution
    '''

    def U(self, x, y):
        return np.exp(x + y)

    def dUdx(self, x, y):
        return np.exp(x + y)

    def dUdy(self, x, y):
        return np.exp(x + y)

    def G(self, x, y):
        '''
        Dirichlet boundary condition
        '''
        if x == -1:
            return np.exp(-1 + y)
        if x == 1:
            return np.exp(1 + y)
        if y == 1:
            return np.exp(1 + x)
        return 'No limit'

    def dGQbPb(self, x, y):
        '''
        Robin boundary condition
        dU * n +ru = q
        '''
        r = 1
        q = 0
        if y == -1:
            return r, q
        else:
            return 'No limit'


def example1():
    print('       %40s' % ('example1--Dirichlet--One_Dimensional_Linear_Element'))
    print('-' * 72)
    print('%2s     %11s      %10s   %10s    %10s      time' % ('h', 'maxerror', '||U-Uh||oo', '||U-Uh||0', '||U-Uh||1'))
    errors = [[], [], [], []]
    for i in range(2, 8):
        start = time.time()
        h = 1 / (2 ** i)
        question = Question_Dirichlet()

        solver = Solver(question=question, h_Mesh=h, h_Finite=h)
        solver.setBasisFunction(trialBasisFunctionType=BasisFunctionType.One_Dimensional_Linear_Element,
                                testBasisFunctionType=BasisFunctionType.One_Dimensional_Linear_Element)
        solver.setIntegrator(integratorType=IntegratorType.Gauss)
        solver.integrator.set_gauss_point_number(4)
        solver.setPT_PbTb(PMatrixType.One_Dimensional_Linear_Node, TMatrixType.One_Dimensional_Linear_Cell,
                          PMatrixType.One_Dimensional_Linear_Node, TMatrixType.One_Dimensional_Linear_Cell)

        solver.makeLinearSystem()
        solver.makeBoundaryProcess()
        solver.solve()

        end = time.time()
        evaluate = Evaluate(solver)
        print('1/%-3s    %.4e     %.4e    %.4e    %.4e    %.4f' % (
            str(2 ** i), evaluate.get_maximum_error(), evaluate.Loo_norm_error(), evaluate.L2_norm_error(),
            evaluate.H1_semi_norm_error(), end - start))

        errors[0].append(evaluate.get_maximum_error())
        errors[1].append(evaluate.Loo_norm_error())
        errors[2].append(evaluate.L2_norm_error())
        errors[3].append(evaluate.H1_semi_norm_error())
    start = time.time()
    orders = Evaluate.get_convergence_order(errors)
    end = time.time()
    print(' O         O(h^%d)         O(h^%d)        O(h^%d)        O(h^%d)      %.4f' % (
        orders[0], orders[1], orders[2], orders[3], end - start))
    print('-' * 72)


def example2():
    print('       %40s' % ('example2--Dirichlet--One_Dimensional_Quadratic_Element'))
    print('-' * 72)
    print('%2s     %11s      %10s   %10s    %10s      time' % ('h', 'maxerror', '||U-Uh||oo', '||U-Uh||0', '||U-Uh||1'))
    errors = [[], [], [], []]
    for i in range(2, 8):
        start = time.time()
        h = 1 / (2 ** i)
        question = Question_Dirichlet()

        solver = Solver(question=question, h_Mesh=h, h_Finite=h / 2)
        solver.setBasisFunction(trialBasisFunctionType=BasisFunctionType.One_Dimensional_Quadratic_Element,
                                testBasisFunctionType=BasisFunctionType.One_Dimensional_Quadratic_Element)
        solver.setIntegrator(IntegratorType.Gauss)
        solver.integrator.set_gauss_point_number(4)
        solver.setPT_PbTb(PMatrixType.One_Dimensional_Linear_Node, TMatrixType.One_Dimensional_Linear_Cell,
                          PMatrixType.One_Dimensional_Linear_Node, TMatrixType.One_Dimensional_Quadratic_Cell)

        solver.makeLinearSystem()
        solver.makeBoundaryProcess()
        solver.solve()

        end = time.time()
        evaluate = Evaluate(solver)
        print('1/%-3s    %.4e     %.4e    %.4e    %.4e    %.4f' % (
            str(2 ** i), evaluate.get_maximum_error(), evaluate.Loo_norm_error(), evaluate.L2_norm_error(),
            evaluate.H1_semi_norm_error(), end - start))
        errors[0].append(evaluate.get_maximum_error())
        errors[1].append(evaluate.Loo_norm_error())
        errors[2].append(evaluate.L2_norm_error())
        errors[3].append(evaluate.H1_semi_norm_error())
    start = time.time()
    orders = Evaluate.get_convergence_order(errors)
    end = time.time()
    print(' O         O(h^%d)         O(h^%d)        O(h^%d)        O(h^%d)      %.4f' % (
        orders[0], orders[1], orders[2], orders[3], end - start))
    print('-' * 72)


def example3():
    print('       %40s' % ('example3--Neumann--One_Dimensional_Quadratic_Element'))
    print('-' * 72)
    print('%2s     %11s      %10s   %10s    %10s      time' % ('h', 'maxerror', '||U-Uh||oo', '||U-Uh||0', '||U-Uh||1'))
    errors = [[], [], [], []]
    for i in range(2, 8):
        start = time.time()
        h = 1 / (2 ** i)
        question = Question_Neumann()

        solver = Solver(question=question, h_Mesh=h, h_Finite=h / 2)
        solver.setBasisFunction(trialBasisFunctionType=BasisFunctionType.One_Dimensional_Quadratic_Element,
                                testBasisFunctionType=BasisFunctionType.One_Dimensional_Quadratic_Element)
        solver.setIntegrator(IntegratorType.Gauss)
        solver.integrator.set_gauss_point_number(4)
        solver.setPT_PbTb(PMatrixType.One_Dimensional_Linear_Node, TMatrixType.One_Dimensional_Linear_Cell,
                          PMatrixType.One_Dimensional_Linear_Node, TMatrixType.One_Dimensional_Quadratic_Cell)

        solver.makeLinearSystem()
        solver.makeBoundaryProcess()
        solver.solve()

        end = time.time()
        evaluate = Evaluate(solver)
        print('1/%-3s    %.4e     %.4e    %.4e    %.4e    %.4f' % (
            str(2 ** i), evaluate.get_maximum_error(), evaluate.Loo_norm_error(), evaluate.L2_norm_error(),
            evaluate.H1_semi_norm_error(), end - start))
        errors[0].append(evaluate.get_maximum_error())
        errors[1].append(evaluate.Loo_norm_error())
        errors[2].append(evaluate.L2_norm_error())
        errors[3].append(evaluate.H1_semi_norm_error())
    start = time.time()
    orders = Evaluate.get_convergence_order(errors)
    end = time.time()
    print(' O         O(h^%d)         O(h^%d)        O(h^%d)        O(h^%d)      %.4f' % (
        orders[0], orders[1], orders[2], orders[3], end - start))
    print('-' * 72)


def example4():
    print('       %40s' % ('example4--Robin--One_Dimensional_Quadratic_Element'))
    print('-' * 72)
    print('%2s     %11s      %10s   %10s    %10s      time' % ('h', 'maxerror', '||U-Uh||oo', '||U-Uh||0', '||U-Uh||1'))
    errors = [[], [], [], []]
    for i in range(2, 8):
        start = time.time()
        h = 1 / (2 ** i)
        question = Question_Robin()

        solver = Solver(question=question, h_Mesh=h, h_Finite=h / 2)
        solver.setBasisFunction(trialBasisFunctionType=BasisFunctionType.One_Dimensional_Quadratic_Element,
                                testBasisFunctionType=BasisFunctionType.One_Dimensional_Quadratic_Element)
        solver.setIntegrator(IntegratorType.Gauss)
        solver.integrator.set_gauss_point_number(4)
        solver.setPT_PbTb(PMatrixType.One_Dimensional_Linear_Node, TMatrixType.One_Dimensional_Linear_Cell,
                          PMatrixType.One_Dimensional_Linear_Node, TMatrixType.One_Dimensional_Quadratic_Cell)

        solver.makeLinearSystem()
        solver.makeBoundaryProcess()
        solver.solve()

        end = time.time()
        evaluate = Evaluate(solver)
        print('1/%-3s    %.4e     %.4e    %.4e    %.4e    %.4f' % (
            str(2 ** i), evaluate.get_maximum_error(), evaluate.Loo_norm_error(), evaluate.L2_norm_error(),
            evaluate.H1_semi_norm_error(), end - start))
        errors[0].append(evaluate.get_maximum_error())
        errors[1].append(evaluate.Loo_norm_error())
        errors[2].append(evaluate.L2_norm_error())
        errors[3].append(evaluate.H1_semi_norm_error())
    start = time.time()
    orders = Evaluate.get_convergence_order(errors)
    end = time.time()
    print(' O         O(h^%d)         O(h^%d)        O(h^%d)        O(h^%d)      %.4f' % (
        orders[0], orders[1], orders[2], orders[3], end - start))
    print('-' * 72)


def example5():
    print('       %40s' % ('example5--Dirichlet--Two_Dimensional_Linear_Element'))
    print('-' * 72)
    print('%2s     %11s      %10s   %10s    %10s      time' % ('h', 'maxerror', '||U-Uh||oo', '||U-Uh||0', '||U-Uh||1'))
    errors = [[], [], [], []]
    for i in range(2, 4):
        start = time.time()
        h = 1 / (2 ** i)
        question = Question_Dirichlet_2D()

        solver = Solver(question=question, h_Mesh=(h, h), h_Finite=(h, h))
        solver.setBasisFunction(trialBasisFunctionType=BasisFunctionType.Two_Dimensional_Linear_Element,
                                testBasisFunctionType=BasisFunctionType.Two_Dimensional_Linear_Element)
        solver.setIntegrator(IntegratorType.Gauss)
        solver.integrator.set_gauss_point_number(9)
        solver.setPT_PbTb(PMatrixType.Two_Dimensional_Linear_Node, TMatrixType.Two_Dimensional_Linear_Cell,
                          PMatrixType.Two_Dimensional_Linear_Node, TMatrixType.Two_Dimensional_Linear_Cell)
        solver.makeLinearSystem()
        solver.makeBoundaryProcess()
        solver.solve()
        end = time.time()
        evaluate = Evaluate(solver)
        print('1/%-3s    %.4e     %.4e    %.4e    %.4e    %.4f' % (
            str(2 ** i), evaluate.get_maximum_error(), evaluate.Loo_norm_error(), evaluate.L2_norm_error(),
            evaluate.H1_semi_norm_error(), end - start))

        errors[0].append(evaluate.get_maximum_error())
        errors[1].append(evaluate.Loo_norm_error())
        errors[2].append(evaluate.L2_norm_error())
        errors[3].append(evaluate.H1_semi_norm_error())
    start = time.time()
    orders = Evaluate.get_convergence_order(errors)
    end = time.time()
    print(' O         O(h^%d)         O(h^%d)        O(h^%d)        O(h^%d)      %.4f' % (
        orders[0], orders[1], orders[2], orders[3], end - start))
    print('-' * 72)


def example6():
    print('       %40s' % ('example6--Dirichlet--Two_Dimensional_Quadratic_Element'))
    print('-' * 72)
    print('%2s     %11s      %10s   %10s    %10s      time' % ('h', 'maxerror', '||U-Uh||oo', '||U-Uh||0', '||U-Uh||1'))
    errors = [[], [], [], []]
    for i in range(2, 4):
        start = time.time()
        h = 1 / (2 ** i)
        question = Question_Dirichlet_2D()
        solver = Solver(question=question, h_Mesh=(h, h), h_Finite=(h / 2, h / 2))
        solver.setBasisFunction(trialBasisFunctionType=BasisFunctionType.Two_Dimensional_Quadratic_Element,
                                testBasisFunctionType=BasisFunctionType.Two_Dimensional_Quadratic_Element)
        solver.setIntegrator(IntegratorType.Gauss)
        solver.integrator.set_gauss_point_number(9)
        solver.setPT_PbTb(PMatrixType.Two_Dimensional_Linear_Node, TMatrixType.Two_Dimensional_Linear_Cell,
                          PMatrixType.Two_Dimensional_Linear_Node, TMatrixType.Two_Dimensional_Quadratic_Cell)
        solver.makeLinearSystem()
        solver.makeBoundaryProcess()
        solver.solve()

        end = time.time()
        evaluate = Evaluate(solver)
        print('1/%-3s    %.4e     %.4e    %.4e    %.4e    %.4f' % (
            str(2 ** i), evaluate.get_maximum_error(), evaluate.Loo_norm_error(), evaluate.L2_norm_error(),
            evaluate.H1_semi_norm_error(), end - start))

        errors[0].append(evaluate.get_maximum_error())
        errors[1].append(evaluate.Loo_norm_error())
        errors[2].append(evaluate.L2_norm_error())
        errors[3].append(evaluate.H1_semi_norm_error())
    start = time.time()
    orders = Evaluate.get_convergence_order(errors)
    end = time.time()
    print(' O         O(h^%d)         O(h^%d)        O(h^%d)        O(h^%d)      %.4f' % (
        orders[0], orders[1], orders[2], orders[3], end - start))
    print('-' * 72)


def example7():
    print('       %40s' % ('example7--Neumann--Two_Dimensional_linear_Element'))
    print('-' * 72)
    print('%2s     %11s      %10s   %10s    %10s      time' % ('h', 'maxerror', '||U-Uh||oo', '||U-Uh||0', '||U-Uh||1'))
    errors = [[], [], [], []]
    for i in range(2, 4):
        start = time.time()
        h = 1 / (2 ** i)
        question = Question_Neumann_2D()
        solver = Solver(question=question, h_Mesh=(h, h), h_Finite=(h, h))
        solver.setBasisFunction(trialBasisFunctionType=BasisFunctionType.Two_Dimensional_Linear_Element,
                                testBasisFunctionType=BasisFunctionType.Two_Dimensional_Linear_Element)
        solver.setIntegrator(IntegratorType.Gauss)
        solver.integrator.set_gauss_point_number(9)
        solver.setPT_PbTb(PMatrixType.Two_Dimensional_Linear_Node, TMatrixType.Two_Dimensional_Linear_Cell,
                          PMatrixType.Two_Dimensional_Linear_Node, TMatrixType.Two_Dimensional_Linear_Cell)

        solver.makeLinearSystem()

        solver.makeBoundaryProcess()
        solver.solve()

        end = time.time()
        evaluate = Evaluate(solver)
        print('1/%-3s    %.4e     %.4e    %.4e    %.4e    %.4f' % (
            str(2 ** i), evaluate.get_maximum_error(), evaluate.Loo_norm_error(), evaluate.L2_norm_error(),
            evaluate.H1_semi_norm_error(), end - start))

        errors[0].append(evaluate.get_maximum_error())
        errors[1].append(evaluate.Loo_norm_error())
        errors[2].append(evaluate.L2_norm_error())
        errors[3].append(evaluate.H1_semi_norm_error())
    start = time.time()
    orders = Evaluate.get_convergence_order(errors)
    end = time.time()
    print(' O         O(h^%d)         O(h^%d)        O(h^%d)        O(h^%d)      %.4f' % (
        orders[0], orders[1], orders[2], orders[3], end - start))
    print('-' * 72)


def example8():
    print('       %40s' % ('example8--Neumann--Two_Dimensional_Quadratic_Element'))
    print('-' * 72)
    print('%2s     %11s      %10s   %10s    %10s      time' % ('h', 'maxerror', '||U-Uh||oo', '||U-Uh||0', '||U-Uh||1'))
    errors = [[], [], [], []]
    for i in range(2, 4):
        start = time.time()
        h = 1 / (2 ** i)
        question = Question_Neumann_2D()
        solver = Solver(question=question, h_Mesh=(h, h), h_Finite=(h / 2, h / 2))
        solver.setBasisFunction(trialBasisFunctionType=BasisFunctionType.Two_Dimensional_Quadratic_Element,
                                testBasisFunctionType=BasisFunctionType.Two_Dimensional_Quadratic_Element)
        solver.setIntegrator(IntegratorType.Gauss)
        solver.integrator.set_gauss_point_number(9)
        solver.setPT_PbTb(PMatrixType.Two_Dimensional_Linear_Node, TMatrixType.Two_Dimensional_Linear_Cell,
                          PMatrixType.Two_Dimensional_Linear_Node, TMatrixType.Two_Dimensional_Quadratic_Cell)

        solver.makeLinearSystem()

        solver.makeBoundaryProcess()
        solver.solve()

        end = time.time()
        evaluate = Evaluate(solver)
        print('1/%-3s    %.4e     %.4e    %.4e    %.4e    %.4f' % (
            str(2 ** i), evaluate.get_maximum_error(), evaluate.Loo_norm_error(), evaluate.L2_norm_error(),
            evaluate.H1_semi_norm_error(), end - start))

        errors[0].append(evaluate.get_maximum_error())
        errors[1].append(evaluate.Loo_norm_error())
        errors[2].append(evaluate.L2_norm_error())
        errors[3].append(evaluate.H1_semi_norm_error())
    start = time.time()
    orders = Evaluate.get_convergence_order(errors)
    end = time.time()
    print(' O         O(h^%d)         O(h^%d)        O(h^%d)        O(h^%d)      %.4f' % (
        orders[0], orders[1], orders[2], orders[3], end - start))
    print('-' * 72)


def example9():
    print('       %40s' % ('example9--Robin--Two_Dimensional_linear_Element'))
    print('-' * 72)
    print('%2s     %11s      %10s   %10s    %10s      time' % ('h', 'maxerror', '||U-Uh||oo', '||U-Uh||0', '||U-Uh||1'))
    errors = [[], [], [], []]
    for i in range(2, 4):
        start = time.time()
        h = 1 / (2 ** i)
        question = Question_Robin_2D()
        solver = Solver(question=question, h_Mesh=(h, h), h_Finite=(h, h))
        solver.setBasisFunction(trialBasisFunctionType=BasisFunctionType.Two_Dimensional_Linear_Element,
                                testBasisFunctionType=BasisFunctionType.Two_Dimensional_Linear_Element)
        solver.setIntegrator(IntegratorType.Gauss)
        solver.integrator.set_gauss_point_number(9)
        solver.setPT_PbTb(PMatrixType.Two_Dimensional_Linear_Node, TMatrixType.Two_Dimensional_Linear_Cell,
                          PMatrixType.Two_Dimensional_Linear_Node, TMatrixType.Two_Dimensional_Linear_Cell)

        solver.makeLinearSystem()
        solver.makeBoundaryProcess()
        solver.solve()

        end = time.time()
        evaluate = Evaluate(solver)
        print('1/%-3s    %.4e     %.4e    %.4e    %.4e    %.4f' % (
            str(2 ** i), evaluate.get_maximum_error(), evaluate.Loo_norm_error(), evaluate.L2_norm_error(),
            evaluate.H1_semi_norm_error(), end - start))
        errors[0].append(evaluate.get_maximum_error())
        errors[1].append(evaluate.Loo_norm_error())
        errors[2].append(evaluate.L2_norm_error())
        errors[3].append(evaluate.H1_semi_norm_error())
    start = time.time()
    orders = Evaluate.get_convergence_order(errors)
    end = time.time()
    print(' O         O(h^%d)         O(h^%d)        O(h^%d)        O(h^%d)      %.4f' % (
        orders[0], orders[1], orders[2], orders[3], end - start))
    print('-' * 72)


def example10():
    print('       %40s' % ('example10--Robin--Two_Dimensional_Quadratic_Element'))
    print('-' * 72)
    print('%2s     %11s      %10s   %10s    %10s      time' % ('h', 'maxerror', '||U-Uh||oo', '||U-Uh||0', '||U-Uh||1'))
    errors = [[], [], [], []]
    for i in range(2, 4):
        start = time.time()
        h = 1 / (2 ** i)
        question = Question_Robin_2D()
        solver = Solver(question=question, h_Mesh=(h, h), h_Finite=(h / 2, h / 2))
        solver.setBasisFunction(trialBasisFunctionType=BasisFunctionType.Two_Dimensional_Quadratic_Element,
                                testBasisFunctionType=BasisFunctionType.Two_Dimensional_Quadratic_Element)
        solver.setIntegrator(IntegratorType.Gauss)
        solver.integrator.set_gauss_point_number(9)
        solver.setPT_PbTb(PMatrixType.Two_Dimensional_Linear_Node, TMatrixType.Two_Dimensional_Linear_Cell,
                          PMatrixType.Two_Dimensional_Linear_Node, TMatrixType.Two_Dimensional_Quadratic_Cell)

        solver.makeLinearSystem()

        solver.makeBoundaryProcess()
        solver.solve()

        end = time.time()
        evaluate = Evaluate(solver)
        print('1/%-3s    %.4e     %.4e    %.4e    %.4e    %.4f' % (
            str(2 ** i), evaluate.get_maximum_error(), evaluate.Loo_norm_error(), evaluate.L2_norm_error(),
            evaluate.H1_semi_norm_error(), end - start))

        errors[0].append(evaluate.get_maximum_error())
        errors[1].append(evaluate.Loo_norm_error())
        errors[2].append(evaluate.L2_norm_error())
        errors[3].append(evaluate.H1_semi_norm_error())
    start = time.time()
    orders = Evaluate.get_convergence_order(errors)
    end = time.time()
    print(' O         O(h^%d)         O(h^%d)        O(h^%d)        O(h^%d)      %.4f' % (
        orders[0], orders[1], orders[2], orders[3], end - start))
    print('-' * 72)


if __name__ == '__main__':
    example1()
    example2()
    example3()
    example4()
    example5()
    example6()
    example7()
    example8()
    example9()
    example10()
