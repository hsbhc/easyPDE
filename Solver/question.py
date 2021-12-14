import numpy as np

'''
 equation : - d/dx (C(x)* (dU(x)/dx)) = F(x)  a <= x <=b
 U(a) = 0  U(b) = cos(1)  du/dx|b= cos(1)-sin(1)
'''
class Question():
    def __init__(self,range=(0,1)):
        self.range=range
        self.rs_left=(1,1)
        self.rs_right=(0,)

    def C(self,x):
        return np.exp(x)
    def F(self,x):
        return -np.exp(x) * (np.cos(x) - 2 * np.sin(x) - x * np.cos(x) - x * np.sin(x))

    #dirichlet
    def G(self,x):
        if x == 0:
            return 0
        if x == 1:
            return np.cos(1)
        return 'No limit'

    # #neumann
    # def G(self,x):
    #     if x == 0:
    #         return 0
    #     else:
    #         return 'No limit'
    #
    # def dG(self,x):
    #     if x == 1:
    #         return np.cos(1)-np.sin(1)
    #     else:
    #         return 'No limit'

    # #robin
    # def G(self, x):
    #     if x == 1:
    #         return np.cos(1)
    #     else:
    #         return 'No limit'
    # def dGQbPb(self,x):
    #     Q=1
    #     P=1
    #     if x == 0:
    #         return Q,P
    #     else:
    #         return 'No limit'

#answer
def U(x):
    return x*np.cos(x)

def dU(x):
    return np.cos(x)-x*np.sin(x)






