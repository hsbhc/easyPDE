#-- coding: utf-8 --
'''
@Project: easyPDE
@File: basisFunction.py
@Author: SryM.HJ
@Ide: PyCharm
@Time: 2021/12/6 19:11
@Function: Local basis function library
'''
from enum import Enum

class BasisFunctionType(Enum):
    One_Dimensional_Linear_Element = 1
    One_Dimensional_Quadratic_Element=2

class BasisFunction():
    def __init__(self):
        self._trial=self._one_dimensional_linear_element
        self._test=self._one_dimensional_linear_element
        self._Nlb_trial=2
        self._Nlb_test=2

    def _one_dimensional_linear_element(self, x, parameter, index, derivative=0):
        '''
        One dimensional linear element
        :param x: Input
        :param parameter: En such as [Xn,Xn+1]
        :param index: alpha or beta
        :param derivative: Order of derivative
        :return: Basis function value
        '''
        h = parameter[1] - parameter[0]
        if derivative == 0:
            if index == 0:
                return (parameter[1] - x) / h
            if index == 1:
                return (x - parameter[0]) / h
        if derivative == 1:
            if index == 0:
                return -1 / h
            if index == 1:
                return 1 / h

    def _one_dimensional_quadratic_element(self, x, parameter, index, derivative=0):
        '''
        one_dimensional_quadratic_element
        :param x: Input
        :param parameter: En such as [Xn,Xn+1]
        :param index: alpha or beta
        :param derivative: Order of derivative
        :return: Basis function value
        '''
        h = parameter[1] - parameter[0]
        if derivative == 0:
            if index == 0:
                return  2*(((x-parameter[0])/h)**2)-3*((x-parameter[0])/h)+1
            if index == 1:
                return 2*(((x-parameter[0])/h)**2)-((x-parameter[0])/h)
            if index == 2:
                return -4*(((x-parameter[0])/h)**2)+4*((x-parameter[0])/h)
        if derivative == 1:
            if index == 0:
                return 4*((x-parameter[0])/h)*(1/h)-3/h
            if index == 1:
                return 4*((x-parameter[0])/h)*(1/h)-1/h
            if index == 2:
                return -8*((x-parameter[0])/h)*(1/h)+4/h


    def get_trial_test(self,trialBasisFunctionType:BasisFunctionType,testBasisFunctionType:BasisFunctionType):
        '''
        gain trialBasisFunction and testBasisFunction according to type
        :param trialBasisFunctionType: @see BasisFunction
        :param testBasisFunctionType: @see BasisFunction
        :return: Nlb_trial,trialBasisFunction,Nlb_test,testBasisFunction
        '''
        self._Nlb_trial,self._trial=self._getBasisFunction(trialBasisFunctionType)
        self._Nlb_test,self._test=self._getBasisFunction(testBasisFunctionType)
        return self._Nlb_trial,self._trial,self._Nlb_test,self._test

    def _getBasisFunction(self,basisFunctionType:BasisFunctionType):
        '''
        gain BasisFunction according to type
        :param basisFunctionType: @see BasisFunction
        :return: Nlb_basisFunction,basisFunction
        '''
        if basisFunctionType == BasisFunctionType.One_Dimensional_Linear_Element:
            print(0)
            return 2 , self._one_dimensional_linear_element

        if basisFunctionType == BasisFunctionType.One_Dimensional_Quadratic_Element:
            return 3, self._one_dimensional_quadratic_element
