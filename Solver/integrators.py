import numpy as np
from enum import Enum
from abc import ABCMeta,abstractmethod

class IntegratorType(Enum):
    Gauss = 1

class Integrator(metaclass=ABCMeta):
    @abstractmethod
    def integral(self,a,b,f):
        raise NotImplementedError

class GaussIntegrator(Integrator):
    gauss_point_number: int
    def __init__(self):
        self.gauss_point_number=4
        self.w,self.t=self._generate_Gauss_point_w(self.gauss_point_number)
    def set_gauss_point_number(self,gauss_point_number):
        self.gauss_point_number = gauss_point_number
        self.w, self.t = self._generate_Gauss_point_w(self.gauss_point_number)
    def _generate_Gauss_point_w(self,gauss_point_number):
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
        w = ((b - a) / 2) * self.w
        x = (b + a) / 2 + ((b - a) / 2) * self.t
        result = 0.0
        for i in range(self.gauss_point_number):
            result += w[i] * f(x[i])
        return result

    def getX(self,a,b):
        x = (b + a) / 2 + ((b - a) / 2) * self.t
        return x
def getIntegrator(type:IntegratorType):
    if type==IntegratorType.Gauss:
        return GaussIntegrator()
