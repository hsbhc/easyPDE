#-- coding: utf-8 --
'''
@Project: easyPDE
@File: integrandFunctions.py
@Author: SryM.HJ
@Ide: PyCharm
@Time: 2021/12/6 19:51
@Function: The left and right integrands of mesh elements in the weak form of one-dimensional Poisson equation are defined
'''

from finiteElement1D.question import Question

class IntegrandFunction():
    def __init__(self,question:Question,trial,test):
        '''
        Initialize integrand function
        rs_left: derivative order of left basis function of equation in weak form
        rs_left: derivative order of right basis function of equation in weak form
        a: Lower integral limit
        b: Integral upper limit
        alpha: Index of local trial basis function
        beta: Index of local test basis function
        :param question: The equation
        :param trial: Local trial basis function
        :param test: Local test basis function
        '''
        self._question = question
        self._rs_left=question.rs_left
        self._rs_right= question.rs_right
        self._trial=trial
        self._test=test
        self._a=0
        self._b=0
        self._alpha=0
        self._beta=0

    def set(self,a,b,alpha,beta):
        '''
        Each mesh cell needs to update the basis function
        :param a: Lower integral limit
        :param b: Integral upper limit
        :param alpha: Index of local trial basis function
        :param beta: Index of local test basis function
        :return:
        '''
        self._a=a
        self._b=b
        self._alpha=alpha
        self._beta=beta

    def A_F(self,x):
        '''
        The left integrands of mesh elements in the weak form in current interval [self.a,self.b]
        :param x: Input
        :return: Function value
        '''
        trial=self._trial(x, (self._a, self._b), index=self._alpha, derivative=self._rs_left[0])
        test=self._test(x, (self._a, self._b), index=self._beta, derivative=self._rs_left[1])
        f = self._question.C(x) * trial*test
        return f

    def B_F(self,x):
        '''
        The right integrands of mesh elements in the weak form in current interval [self.a,self.b]
        :param x: Input
        :return: Function value
        '''
        test = self._test(x, (self._a, self._b), index=self._beta, derivative=self._rs_right[0])
        f = self._question.F(x)  * test
        return f
