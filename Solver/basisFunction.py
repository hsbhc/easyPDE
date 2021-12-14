from enum import Enum


class BasisFunctionType(Enum):
    One_Dimensional_Linear_Element = 1
    One_Dimensional_Quadratic_Element=2

class BasisFunction():
    def __init__(self):
        self.trial=self._one_dimensional_linear_element
        self.test=self._one_dimensional_linear_element
        self.Nlb_trial=2
        self.Nlb_test=2

    def _one_dimensional_linear_element(self, x, parameter, index, derivative=0):
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
        self.Nlb_trial,self.trial=self._getBasisFunction(trialBasisFunctionType)
        self.Nlb_test,self.test=self._getBasisFunction(testBasisFunctionType)
        return self.Nlb_trial,self.trial,self.Nlb_test,self.test

    def _getBasisFunction(self,basisFunctionType:BasisFunctionType):
        if basisFunctionType == BasisFunctionType.One_Dimensional_Linear_Element:
            return 2 , self._one_dimensional_linear_element

        if basisFunctionType == BasisFunctionType.One_Dimensional_Quadratic_Element:
            return 3, self._one_dimensional_quadratic_element