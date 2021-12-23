#-- coding: utf-8 --
'''
@Project: easyPDE
@File: integrandFunctions.py
@Author: SryM.HJ
@Ide: PyCharm
@Time: 2021/12/6 19:51
@Function: The left and right integrands of mesh elements in the weak form of one-dimensional Poisson equation are defined
'''

from finiteElement2D_t.question import Question, One_Dimensional_PoissonQuestion, Two_Dimensional_PoissonQuestion_t


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
        self._rs_left=question.rs_left  #[r,s]
        self._rs_right= question.rs_right #[r,s]
        self._trial=trial
        self._test=test
        self.En=None
        self._alpha=0
        self._beta=0

    def set(self,En,alpha,beta):
        '''
        Each mesh cell needs to update the basis function
        :param a: Lower integral limit
        :param b: Integral upper limit
        :param alpha: Index of local trial basis function
        :param beta: Index of local test basis function
        :return:
        '''
        self.En=En.correct_En
        self._alpha=alpha
        self._beta=beta

    def A_F(self,x):
        '''
        The left integrands of mesh elements in the weak form in current interval [self.a,self.b]
        :param x: Input
        :return: Function value
        '''
        a = self.En[0][0]
        b = self.En[1][0]
        trial=self._trial(x, (a, b), index=self._alpha, derivative=self._rs_left[0])
        test=self._test(x, (a, b), index=self._beta, derivative=self._rs_left[1])
        f = self._question.C(x) * trial*test
        return f

    def B_F(self,x):
        '''
        The right integrands of mesh elements in the weak form in current interval [self.a,self.b]
        :param x: Input
        :return: Function value
        '''
        a = self.En[0][0]
        b = self.En[1][0]
        test = self._test(x, (a, b), index=self._beta, derivative=self._rs_right[0])
        f = self._question.F(x)  * test
        return f

class IntegrandFunction2D():
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
        self._rs_left=question.rs_left  # [(r,s,p,q),(r,s,p,q)]
        self._pq_right= question.pq_right #[(p,q),(p,q)]
        self.trial=trial
        self.test=test
        self.En=None
        self._alpha=0
        self._beta=0

    def set(self,En,alpha,beta):
        '''
        Each mesh cell needs to update the basis function
        :param a: Lower integral limit
        :param b: Integral upper limit
        :param alpha: Index of local trial basis function
        :param beta: Index of local test basis function
        :return:
        '''
        self.En=En.correct_En
        self._alpha=alpha
        self._beta=beta

    def A_F(self,x,y):
        '''
        The left integrands of mesh elements in the weak form in current interval [self.a,self.b]
        :param x: Input
        :return: Function value
        '''
        f=0.0
        for r,s,p,q in self._rs_left:
            trial=self.trial(x,y, self.En, index=self._alpha, derivative_x=r,derivative_y=s)
            test=self.test(x,y, self.En, index=self._beta, derivative_x=p,derivative_y=q)
            f_l=self._question.C(x, y)* trial*test
            f = f+f_l
        return f

    def B_F(self,x,y):
        '''
        The right integrands of mesh elements in the weak form in current interval [self.a,self.b]
        :param x: Input
        :return: Function value
        '''
        f = 0.0
        for p,q in self._pq_right:
            test = self.test(x,y,self.En, index=self._beta, derivative_x=p,derivative_y=q)
            f_l = self._question.F(x, y)* test
            f = f + f_l
        return f


class IntegrandFunction2D_t():
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
        self._rs_left=question.rs_left  # [(r,s,p,q),(r,s,p,q)]
        self._pq_right= question.pq_right #[(p,q),(p,q)]
        self.trial=trial
        self.test=test
        self.En=None
        self._alpha=0
        self._beta=0

        self.t=0
    def set(self,En,alpha,beta):
        '''
        Each mesh cell needs to update the basis function
        :param a: Lower integral limit
        :param b: Integral upper limit
        :param alpha: Index of local trial basis function
        :param beta: Index of local test basis function
        :return:
        '''
        self.En=En.correct_En
        self._alpha=alpha
        self._beta=beta

    def A_F(self,x,y):
        '''
        The left integrands of mesh elements in the weak form in current interval [self.a,self.b]
        :param x: Input
        :return: Function value
        '''
        f=0.0
        for r,s,p,q in self._rs_left:
            trial=self.trial(x,y, self.En, index=self._alpha, derivative_x=r,derivative_y=s)
            test=self.test(x,y, self.En, index=self._beta, derivative_x=p,derivative_y=q)
            f_l=self._question.C(x, y,self.t)* trial*test
            f = f+f_l
        return f

    def M_F(self,x,y):
        '''
        The left integrands of mesh elements in the weak form in current interval [self.a,self.b]
        :param x: Input
        :return: Function value
        '''
        f=0.0
        trial=self.trial(x,y, self.En, index=self._alpha, derivative_x=0,derivative_y=0)
        test=self.test(x,y, self.En, index=self._beta, derivative_x=0,derivative_y=0)
        f_l=trial*test
        f = f+f_l
        return f

    def B_F(self,x,y):
        '''
        The right integrands of mesh elements in the weak form in current interval [self.a,self.b]
        :param x: Input
        :return: Function value
        '''
        f = 0.0
        for p,q in self._pq_right:
            test = self.test(x,y,self.En, index=self._beta, derivative_x=p,derivative_y=q)
            f_l = self._question.F(x, y,self.t)* test
            f = f + f_l
        return f


def getIntegrandFunction(question:Question,trial,test):
    if isinstance(question, (One_Dimensional_PoissonQuestion)):
        return IntegrandFunction(question, trial, test)
    elif isinstance(question,(Two_Dimensional_PoissonQuestion_t)):
        return IntegrandFunction2D_t(question,trial,test)
    else:
        return IntegrandFunction2D(question,trial,test)