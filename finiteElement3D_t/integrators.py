#-- coding: utf-8 --
'''
@Project: easyPDE
@File: integrators.py
@Author: SryM.HJ
@Ide: PyCharm
@Time: 2021/12/6 19:24
@Function: Integrator Library
'''
import time

import numpy as np
from enum import Enum
from abc import ABCMeta,abstractmethod
from finiteElement3D_t.PTmatrix import EN


class IntegratorType(Enum):
    Gauss = 1

class Integrator(metaclass=ABCMeta):
    @abstractmethod
    def integral(self,En:EN,f):
        '''
        Integral calculation
        :param En: integral limit
        :param f: Integrand function
        :return: Integral value
        '''
        raise NotImplementedError

class GaussIntegrator(Integrator):
    gauss_point_number: int
    def __init__(self):
        self._gauss_point_number=4
        self._w,self._t=self._generate_Gauss_point_w(self._gauss_point_number)
        self._2dw, self._2dt = self._generate_2DGauss_point_w(self._gauss_point_number)

    def set_gauss_point_number(self,gauss_point_number):
        '''
        Set the Gaussian integration point of the Gaussian integrator according to the number of Gaussian integral points
        :param gauss_point_number: Number of Gauss integral points
        :return: None
        '''
        self._gauss_point_number = gauss_point_number
        self._w, self._t = self._generate_Gauss_point_w(self._gauss_point_number)
        self._2dw, self._2dt = self._generate_2DGauss_point_w(self._gauss_point_number)
        self._3dw,self._3dt=self._generate_3DGauss_point_w(self._gauss_point_number)
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

    def _generate_2DGauss_point_w(self, gauss_point_number):
        w, t = [], []
        if gauss_point_number == 4:
            w = np.array([(1 - 1 / np.sqrt(3)) / 8, (1 - 1 / np.sqrt(3)) / 8, (1 + 1 / np.sqrt(3)) / 8,(1 + 1 / np.sqrt(3)) / 8])
            t =np.array([[(1 / np.sqrt(3) + 1) / 2, (1 - 1 / np.sqrt(3)) * (1 + 1 / np.sqrt(3)) / 4],
                [(1 / np.sqrt(3) + 1) / 2, (1 - 1 / np.sqrt(3)) * (1 - 1 / np.sqrt(3)) / 4],
                [(-1 / np.sqrt(3) + 1) / 2, (1 + 1 / np.sqrt(3)) * (1 + 1 / np.sqrt(3)) / 4],
                [(-1 / np.sqrt(3) + 1) / 2, (1 + 1 / np.sqrt(3)) * (1 - 1 / np.sqrt(3)) / 4]])

        if gauss_point_number  == 9:
            w = np.array([64 / 81 * (1 - 0) / 8, 100 / 324 * (1 - np.sqrt(3 / 5)) / 8,100 / 324 * (1 - np.sqrt(3 / 5)) / 8,100 / 324 * (1 + np.sqrt(3 / 5)) / 8,
                 100 / 324 * (1 + np.sqrt(3 / 5)) / 8, 40 / 81 * (1 - 0) / 8,40 / 81 * (1 - 0) / 8, 40 / 81 * (1 - np.sqrt(3 / 5)) / 8,40 / 81 * (1 + np.sqrt(3 / 5)) / 8])
            t = np.array([[(1 + 0) / 2, (1 - 0) * (1 + 0) / 4],
            [(1 + np.sqrt(3 / 5)) / 2, (1 - np.sqrt(3 / 5)) * (1 + np.sqrt(3 / 5)) / 4],
            [(1 + np.sqrt(3 / 5)) / 2, (1 - np.sqrt(3 / 5)) * (1 - np.sqrt(3 / 5)) / 4],
            [(1 - np.sqrt(3 / 5)) / 2, (1 + np.sqrt(3 / 5)) * (1 +np.sqrt(3 / 5)) / 4],
            [(1 - np.sqrt(3 / 5)) / 2, (1 + np.sqrt(3 / 5)) * (1 - np.sqrt(3 / 5)) / 4],
            [(1 + 0) / 2, (1 - 0) * (1 + np.sqrt(3 / 5)) / 4],
            [(1 + 0) / 2, (1 - 0) * (1 - np.sqrt(3 / 5)) / 4],
            [(1 + np.sqrt(3 / 5)) / 2, (1 - np.sqrt(3 / 5)) * (1 + 0) / 4],
            [(1 - np.sqrt(3 / 5)) / 2, (1 + np.sqrt(3 / 5)) * (1 + 0) / 4]])
        if gauss_point_number  == 3:
            w = np.array([1 / 6, 1 / 6, 1 / 6])
            t = np.array([[1 / 2, 0],[1 / 2, 1 / 2],[0, 1 / 2]])
        return w,t

    def _generate_3DGauss_point_w(self, gauss_point_number):
        w, t = [], []
        root3=1./np.sqrt(3)
        if gauss_point_number == 8:
            w = np.array([1.,1.,1.,1.,1.,1.,1.,1.])
            t =np.array([
                [root3, root3,root3],
                [root3, root3, -root3],
                [root3, -root3, root3],
                [root3,- root3, -root3],
                [-root3, root3, root3],
                [-root3, -root3, root3],
                [-root3, root3, -root3],
                [-root3, -root3, -root3]
            ])

        return w,t

    def integral(self,En, f):
        '''
        Integral calculation
        :param En: integral limit
        :param f: Integrand function
        :return: Integral value
        '''
        if En.dim==1:
            a=En.correct_En[0][0]
            b=En.correct_En[1][0]
            w = ((b - a) / 2) * self._w
            x = (b + a) / 2 + ((b - a) / 2) * self._t
            result = 0.0
            for i in range(self._gauss_point_number):
                result += w[i] * f(x[i])
            return result

        if En.dim==2:
            point1,point2,point3=En.correct_En[0],En.correct_En[1],En.correct_En[2]
            x1,y1=point1
            x2,y2=point2
            x3,y3=point3

            J = abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
            w = self._2dw * J
            x=x1+(x2-x1)*self._2dt[:,0]+(x3 - x1) * self._2dt[:,1]
            y=y1+(y2 - y1) *self._2dt[:, 0]+(y3 - y1) *self._2dt[:, 1]
            result = 0.0
            for i in range(self._gauss_point_number):
                result += w[i] * f(x[i],y[i])
            return result

        if En.dim==3:
            point1,point2,m1,point4,point5=En.correct_En[0],En.correct_En[1],En.correct_En[2],En.correct_En[3],En.correct_En[4]
            h_x = point4[0] - point1[0]
            h_y = point2[1] - point1[1]
            h_z = point5[2] - point1[2]

            x= (h_x/2)*self._3dt[:,0]+h_x/2+point1[0]
            y= (h_y/2)*self._3dt[:,1]+h_y/2+point1[1]
            z = (h_z / 2) * self._3dt[:, 2] + h_z / 2 + point1[2]

            result = 0.0
            for i in range(self._gauss_point_number):
                result += self._3dw[i] * f(x[i],y[i],z[i])
            return result


    def getX(self,En):
        '''
        Obtain the coordinates of Gaussian integral points in [a,b] interval
        :param a: Interval lower bound
        :param b: Interval upper bound
        :return: x: coordinates of Gaussian integral points in [a,b]
        '''
        if En.dim==1:
            a = En.correct_En[0][0]
            b = En.correct_En[1][0]
            x = (b + a) / 2 + ((b - a) / 2) * self._t
            return x
        if En.dim==2:
            point1, point2, point3 = En.correct_En[0], En.correct_En[1], En.correct_En[2]
            x1, y1, = point1
            x2, y2 = point2
            x3, y3 = point3
            x = x1 + (x2 - x1) * self._2dt[:, 0] + (x3 - x1) * self._2dt[:, 1]
            y = y1 + (y2 - y1) * self._2dt[:, 0] + (y3 - y1) * self._2dt[:, 1]
            return x,y

        if En.dim==3:
            point1, point2, m1, point4, point5 = En.correct_En[0], En.correct_En[1], En.correct_En[2], En.correct_En[3], \
                                                 En.correct_En[4]
            h_x = point4[0] - point1[0]
            h_y = point2[1] - point1[1]
            h_z = point5[2] - point1[2]

            x = (h_x / 2) * self._3dt[:, 0] + h_x / 2 + point1[0]
            y = (h_y / 2) * self._3dt[:, 1] + h_y / 2 + point1[1]
            z = (h_z / 2) * self._3dt[:, 2] + h_z / 2 + point1[2]

            return x,y,z

def getIntegrator(type:IntegratorType):
    '''
    Get integrator by type
    :param type: @see IntegratorType
    :return: Integrator
    '''
    if type==IntegratorType.Gauss:
        return GaussIntegrator()



# correct_En=[[0,0],[1,0],[0,1]]
# En=EN([correct_En],dim=2)
# En.getEnByIndex(0)
#
# inte=getIntegrator(IntegratorType.Gauss)
# inte.set_gauss_point_number(4)
# f= lambda x,y : 2
#
# result=inte.integral(En,f)
# print(result)
