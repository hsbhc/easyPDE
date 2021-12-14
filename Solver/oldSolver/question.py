import numpy as np

'''
 equation : - d/dx (C(x)* (dU(x)/dx)) = F(x)  a <= x <=b
 U(a) = G(x)  U(b) = G(x)
'''
#equation
x_range= (0,1)
def C(x):
    return np.exp(x)
def F(x):
    return -np.exp(x)*(np.cos(x)-2*np.sin(x)-x*np.cos(x)-x*np.sin(x))
def Ua():
    return 0
def Ub():
    return np.cos(1)

def G(x):
    if x==0:
        return 0
    if x==1:
        return np.cos(1)
    return 'No limit'

#input
def trial(x,parameter,index,derivative=0):
    h=parameter[1]-parameter[0]
    if derivative==0:
        if index==0:
            return (parameter[1]-x)/h
        if index==1:
            return (x-parameter[0])/h
    if derivative==1:
        if index==0:
            return -1/h
        if index==1:
            return 1/h
def test(x,parameter,index,derivative=0):
    h=parameter[1]-parameter[0]
    if derivative==0:
        if index==0:
            return (parameter[1]-x)/h
        if index==1:
            return (x-parameter[0])/h
    if derivative==1:
        if index==0:
            return -1/h
        if index==1:
            return 1/h

#answer
def U(x):
    return x*np.cos(x)







