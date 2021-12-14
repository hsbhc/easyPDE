#-- coding: utf-8 --
'''
@Project: easyPDE
@File: linearSystem.py
@Author: SryM.HJ
@Ide: PyCharm
@Time: 2021/12/6 20:25
@Function: The linear system is generated according to the discretized equation in weak form
'''

import numpy as np
from finiteElement2D.PTmatrix import PTMatrix
from finiteElement2D.integrators import Integrator
class MatrixEquation():
    '''
    matrix equation
    '''
    def __init__(self,A,B):
        self.A=A
        self.B=B

    def solve(self):
        self.solution = np.linalg.solve(self.A, self.B)

class LinearSystem():
    def __init__(self,PT_matrix:PTMatrix,Nlb_trial,Nlb_test,integrator:Integrator,integrandFunction):
        '''
        :param PT_matrix: P, T, Pb, Tb, N, Nm, Nb, Nbm
        :param Nlb_trial: Number of local trial basis functions
        :param Nlb_test: Number of local test basis functions
        :param integrator: Integrator
        :param integrandFunction: IntegrandFunction,the left and right integrands of mesh elements
        '''
        self._PT_matrix = PT_matrix
        self.Nlb_trial=Nlb_trial
        self.Nlb_test=Nlb_test
        self.integrandFunction=integrandFunction
        self.integrator=integrator

    def _make_A(self):
        '''
        Make coefficient matrix
        :return: Ab: coefficient matrix
        '''
        Ab=np.zeros((self._PT_matrix.Nbm,self._PT_matrix.Nbm))
        for i in range(self._PT_matrix.N):
            for alpha  in range(self.Nlb_trial):
                for beta in range(self.Nlb_test):
                    self.integrandFunction.set(self._PT_matrix.En.getEnByIndex(i), alpha, beta)
                    value = self.integrator.integral(self._PT_matrix.En.getEnByIndex(i), self.integrandFunction.A_F)
                    Ab[self._PT_matrix.Tb[beta,i],self._PT_matrix.Tb[alpha,i]]+=value
        return Ab

    def _make_B(self):
        '''
        Make constant term
        :return: Bb: constant term vector
        '''
        Bb = np.zeros(self._PT_matrix.Nbm)
        for i in range(self._PT_matrix.N):
            for beta in range(self.Nlb_test):
                self.integrandFunction.set(self._PT_matrix.En.getEnByIndex(i),None, beta)
                value = self.integrator.integral(self._PT_matrix.En.getEnByIndex(i),self.integrandFunction.B_F)
                Bb[self._PT_matrix.Tb[beta, i]] += value
        return Bb


    def getMatrixEquation(self):
        '''
        The matrix equation is obtained by linear system
        :return: MatrixEquation
        '''

        A = self._make_A()
        B = self._make_B()
        return MatrixEquation(A,B)