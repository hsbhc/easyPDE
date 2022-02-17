#-- coding: utf-8 --
'''
@Project: easyPDE
@File: question.py
@Author: SryM.HJ
@Ide: PyCharm
@Time: 2021/12/6 19:36
@Function: one-dimensional Poisson equation and boundary conditions are defined ,the analytical solution of it is also defined
'''
from abc import abstractmethod


class Question(object):

    def U(self,*args):
        pass
    def dU(self,*args):
        pass

'''
one-dimensional Poisson equation and boundary conditions
equation : - d/dx (C(x)* (dU(x)/dx)) = F(x)  a <= x <=b
'''
class One_Dimensional_PoissonQuestion(Question):
    def __init__(self, range=(0, 1)):
        '''
        rs_left: derivative order of left basis function of equation in weak form
        rs_left: derivative order of right basis function of equation in weak form
        :param range: Variable range
        '''
        self.range=range
        self.rs_left = (1, 1)
        self.rs_right = (0,)

    @abstractmethod
    def C(self,x):
        pass

    @abstractmethod
    def F(self,x):
        pass

    '''
    Analytical solution
    '''

    def U(self,x):
        pass

    def dU(self,x):
        pass



class Two_Dimensional_PoissonQuestion(Question):
    def __init__(self, range=((-1, 1),(-1,1))):
        '''
        rs_left: derivative order of left basis function of equation in weak form
        rs_left: derivative order of right basis function of equation in weak form
        :param range: Variable range
        '''
        self.range=range
        self.rs_left = [(1,0,1,0),(0,1,0,1)]
        self.pq_right = [(0,0)]

    @abstractmethod
    def C(self,x,y):
        pass

    @abstractmethod
    def F(self,x,y):
        pass

    '''
    Analytical solution
    '''

    def U(self,x,y):
        pass

    def dU(self,x,y):
        pass


class Two_Dimensional_PoissonQuestion_t(Question):
    def __init__(self, range=((-1, 1),(-1,1)),t_range=(0,1)):
        '''
        rs_left: derivative order of left basis function of equation in weak form
        rs_left: derivative order of right basis function of equation in weak form
        :param range: Variable range
        '''
        self.range=range
        self.t_range=t_range
        self.rs_left = [(1,0,1,0),(0,1,0,1)]
        self.pq_right = [(0,0)]

    @abstractmethod
    def C(self,x,y ,t):
        pass

    @abstractmethod
    def F(self,x,y ,t):
        pass

    '''
    Analytical solution
    '''

    def U(self,x,y,t):
        pass

    def dU(self,x,y,t):
        pass
