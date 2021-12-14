#-- coding: utf-8 --
'''
@Project: easyPDE
@File: integrators.py
@Author: SryM.HJ
@Ide: PyCharm
@Time: 2021/12/6 19:24
@Function: Integrator Library
'''
import numpy as np
from enum import Enum
from abc import ABCMeta,abstractmethod

class IntegratorType(Enum):
    Gauss = 1

class Integrator(metaclass=ABCMeta):
    @abstractmethod
    def integral(self,a,b,f):
        '''
        Integral calculation
        :param a: Lower integral limit
        :param b: Integral upper limit
        :param f: Integrand function
        :return: Integral value
        '''
        raise NotImplementedError

class GaussIntegrator(Integrator):
    gauss_point_number: int
    def __init__(self):
        self._gauss_point_number=4
        self._w,self._t=self._generate_Gauss_point_w(self._gauss_point_number)

    def set_gauss_point_number(self,gauss_point_number):
        '''
        Set the Gaussian integration point of the Gaussian integrator according to the number of Gaussian integral points
        :param gauss_point_number: Number of Gauss integral points
        :return: None
        '''
        self._gauss_point_number = gauss_point_number
        self._w, self._t = self._generate_Gauss_point_w(self._gauss_point_number)

    def _generate_Gauss_point_w(self,gauss_point_number):
        '''
        Obtain the weight and coordinates of Gaussian integral points
        :param gauss_point_number: Number of Gauss integral points
        :return: Weight and coordinates of Gaussian integral points
        '''
        w, t = [], []
        if gauss_point_number == 4:
            w = np.array([0.3478548451, 0.3478548451, 0.6521451549, 0.6521451549])
            t = np.array([0.8611363116, -0.8611363116, 0.3399810436, -0.3399810436])
        if gauss_point_number == 8:
            w = np.array(
                [0.1012285363, 0.1012285363, 0.2223810345, 0.2223810345, 0.3137066459, 0.3137066459, 0.3626837834,
                 0.3626837834])
            t = np.array(
                [0.9602898565, -0.9602898565, 0.7966664774, -0.7966664774, 0.5255324099, -0.5255324099, 0.1834346425,
                 -0.1834346425])
        if gauss_point_number == 2:
            w = np.array([1, 1])
            t = np.array([-1 / np.sqrt(3), 1 / np.sqrt(3)])
        return w, t

    def integral(self, a, b, f):
        '''
        Integral calculation
        :param a: Lower integral limit
        :param b: Integral upper limit
        :param f: Integrand function
        :return: Integral value
        '''
        w = ((b - a) / 2) * self._w
        x = (b + a) / 2 + ((b - a) / 2) * self._t
        result = 0.0
        for i in range(self._gauss_point_number):
            result += w[i] * f(x[i])
        return result

    def getX(self,a,b):
        '''
        Obtain the coordinates of Gaussian integral points in [a,b] interval
        :param a: Interval lower bound
        :param b: Interval upper bound
        :return: x: coordinates of Gaussian integral points in [a,b]
        '''
        x = (b + a) / 2 + ((b - a) / 2) * self._t
        return x

def getIntegrator(type:IntegratorType):
    '''
    Get integrator by type
    :param type: @see IntegratorType
    :return: Integrator
    '''
    if type==IntegratorType.Gauss:
        return GaussIntegrator()

