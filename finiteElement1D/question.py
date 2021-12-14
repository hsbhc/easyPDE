#-- coding: utf-8 --
'''
@Project: easyPDE
@File: question.py
@Author: SryM.HJ
@Ide: PyCharm
@Time: 2021/12/6 19:36
@Function: one-dimensional Poisson equation and boundary conditions are defined ,the analytical solution of it is also defined
'''
import numpy as np

'''
Analytical solution
'''
def U(x):
    return x*np.cos(x)
def dU(x):
    return np.cos(x)-x*np.sin(x)

'''
one-dimensional Poisson equation and boundary conditions
equation : - d/dx (C(x)* (dU(x)/dx)) = F(x)  a <= x <=b
'''
class Question():
    def __init__(self,range=(0,1)):
        '''
        rs_left: derivative order of left basis function of equation in weak form
        rs_left: derivative order of right basis function of equation in weak form
        :param range: Variable range
        '''
        self.range=range
        self.rs_left=(1,1)
        self.rs_right=(0,)

    def C(self,x):
        return np.exp(x)

    def F(self,x):
        return -np.exp(x) * (np.cos(x) - 2 * np.sin(x) - x * np.cos(x) - x * np.sin(x))


    # def G(self,x):
    #     '''
    #     Dirichlet boundary condition
    #     U(a) = G(a)
    #     U(b) = G(b)
    #     '''
    #     if x == 0:
    #         return 0
    #     if x == 1:
    #         return np.cos(1)
    #     return 'No limit'

    # def G(self,x):
    #     '''
    #     Neumann boundary condition
    #     U(a) = G(a)
    #     '''
    #     if x == 0:
    #         return 0
    #     else:
    #         return 'No limit'
    # def dG(self,x):
    #     '''
    #     Neumann boundary condition
    #     dU(b)/dx = G(b)
    #     '''
    #     if x == 1:
    #         return np.cos(1)-np.sin(1)
    #     else:
    #         return 'No limit'

    # def G(self, x):
    #     '''
    #     Robin boundary condition
    #     U(b) = G(b)
    #     '''
    #     if x == 1:
    #         return np.cos(1)
    #     else:
    #         return 'No limit'
    # def dGQbPb(self,x):
    #     '''
    #     Robin boundary condition
    #     dU(a)/dx + Q*U(b) = P*U(b)
    #     '''
    #     Q=1
    #     P=1
    #     if x == 0:
    #         return Q,P
    #     else:
    #         return 'No limit'




