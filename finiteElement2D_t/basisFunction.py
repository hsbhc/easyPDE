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
    Two_Dimensional_Linear_Element=3
    Two_Dimensional_Quadratic_Element=4
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

    def _two_dimensional_linear_element(self, x,y, parameter, index, derivative_x =0,derivative_y =0):
        '''
        Two dimensional linear element
        :param x: Input
        :param parameter: En such as [(xn1,yn1),(xn2,yn2),(xn3,yn3)]
        :param index: alpha or beta
        :param derivative: Order of derivative
        :return: Basis function value
        '''
        point1=parameter[0]
        point2 = parameter[1]
        point3 = parameter[2]
        J=(point2[0]-point1[0])*(point3[1]-point1[1])-(point3[0]-point1[0])*(point2[1]-point1[1])
        x_h=((point3[1]-point1[1])*(x-point1[0])-(point3[0]-point1[0])*(y-point1[1]))/J
        y_h = (-(point2[1] - point1[1]) * (x - point1[0])+ (point2[0] - point1[0]) * (y - point1[1])) / J
        if derivative_x == 0 and derivative_y == 0:
            return self._two_dimensional_linear_element_reference(x_h, y_h, index, 0, 0)

        if derivative_x == 1 and derivative_y == 0:
            return self._two_dimensional_linear_element_reference(x_h, y_h, index,1, 0)*((point3[1]-point1[1])/J)+\
                    self._two_dimensional_linear_element_reference(x_h, y_h, index, 0, 1) * ((point1[1] - point2[1]) / J)

        if derivative_x == 0 and derivative_y == 1:
            return self._two_dimensional_linear_element_reference(x_h, y_h, index, 1, 0) * (
                        (point1[0]-point3[0]) / J) + \
                   self._two_dimensional_linear_element_reference(x_h, y_h, index, 0, 1) * ((point2[0]-point1[0]) / J)

    def _two_dimensional_quadratic_element(self, x,y, parameter, index, derivative_x =0,derivative_y =0):
        '''
        Two dimensional linear element
        :param x: Input
        :param parameter: En such as [(xn1,yn1),(xn2,yn2),(xn3,yn3)]
        :param index: alpha or beta
        :param derivative: Order of derivative
        :return: Basis function value
        '''
        point1= parameter[0]
        point2 = parameter[1]
        point3 = parameter[2]
        J=(point2[0]-point1[0])*(point3[1]-point1[1])-(point3[0]-point1[0])*(point2[1]-point1[1])
        x_h=((point3[1]-point1[1])*(x-point1[0])-(point3[0]-point1[0])*(y-point1[1]))/J
        y_h = (-(point2[1] - point1[1]) * (x - point1[0])+ (point2[0] - point1[0]) * (y - point1[1])) / J
        if derivative_x == 0 and derivative_y == 0:
            return self._two_dimensional_quadratic_element_reference(x_h, y_h, index, 0, 0)

        if derivative_x == 1 and derivative_y == 0:
            return self._two_dimensional_quadratic_element_reference(x_h, y_h, index,1, 0)*((point3[1]-point1[1])/J)+\
                    self._two_dimensional_quadratic_element_reference(x_h, y_h, index, 0, 1) * ((point1[1] - point2[1]) / J)

        if derivative_x == 0 and derivative_y == 1:
            return self._two_dimensional_quadratic_element_reference(x_h, y_h, index, 1, 0) * (
                        (point1[0]-point3[0]) / J) + \
                   self._two_dimensional_quadratic_element_reference(x_h, y_h, index, 0, 1) * ((point2[0]-point1[0]) / J)

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
            return 2 , self._one_dimensional_linear_element

        if basisFunctionType == BasisFunctionType.One_Dimensional_Quadratic_Element:
            return 3, self._one_dimensional_quadratic_element

        if basisFunctionType == BasisFunctionType.Two_Dimensional_Linear_Element:
            return 3, self._two_dimensional_linear_element

        if basisFunctionType == BasisFunctionType.Two_Dimensional_Quadratic_Element:
            return 6, self._two_dimensional_quadratic_element

    def _two_dimensional_linear_element_reference(self,x_h,y_h,index, derivative_x=0,derivative_y=0):
        if derivative_x==0 and derivative_y == 0:
            if index == 0:
                return -x_h-y_h+1
            if index == 1:
                return x_h
            if index == 2:
                return y_h
            return 0
        if derivative_x==1 and derivative_y == 0:
            if index == 0:
                return -1
            if index == 1:
                return 1
            if index == 2:
                return 0
            return 0

        if derivative_x==0 and derivative_y == 1:
            if index == 0:
                return -1
            if index == 1:
                return 0
            if index == 2:
                return 1
            return 0

    def _two_dimensional_quadratic_element_reference(self,x_h,y_h,index, derivative_x=0,derivative_y=0):
        if derivative_x==0 and derivative_y == 0:
            if index == 0:
                return 2*x_h**2+2*y_h**2+4*x_h*y_h-3*y_h-3*x_h+1
            if index == 1:
                return 2*x_h**2-x_h
            if index == 2:
                return 2*y_h**2-y_h
            if index == 3:
                return -4*x_h**2-4*x_h*y_h+4*x_h
            if index == 4:
                return 4*x_h*y_h
            if index == 5:
                return -4*y_h**2-4*x_h*y_h+4*y_h
            return 0
        if derivative_x==1 and derivative_y == 0:
            if index == 0:
                return 4*x_h+4*y_h-3
            if index == 1:
                return 4*x_h-1
            if index == 2:
                return 0
            if index == 3:
                return -8*x_h-4*y_h+4
            if index == 4:
                return 4*y_h
            if index == 5:
                return -4*y_h
            return 0

        if derivative_x==0 and derivative_y == 1:
            if index == 0:
                return 4*y_h+4*x_h-3
            if index == 1:
                return 0
            if index == 2:
                return 4*y_h-1
            if index == 3:
                return -4*x_h
            if index == 4:
                return 4*x_h
            if index == 5:
                return -8*y_h-4*x_h+4
            return 0

