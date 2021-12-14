import numpy as np
from Solver.oldSolver.question import *
np.set_printoptions(linewidth=np.inf)
'''
    Pb,Tb
'''

def make_Pb():
    Pb=np.zeros(N+1)
    for i in range(N+1):
        Pb[i]=left+i*h
    return Pb
def make_Tb():
    Tb = np.zeros((2, N),dtype=int)
    for i in range(N):
        Tb[0,i]=i
        Tb[1,i]=i+1
    return Tb

class IntegrandFunction():
    def __init__(self,a,b,r,s,alpha,beta):
        self.a=a
        self.b=b
        self.r=r
        self.s=s
        self.alpha=alpha
        self.beta=beta
    def A_F(self,x):
        f = C(x) * trial(x, (self.a, self.b), index=self.alpha, derivative=self.r) * test(x, (self.a, self.b), index=self.beta, derivative=self.s)
        return f
    def B_F(self,x):
        f = F(x)  * test(x, (self.a, self.b), index=self.beta, derivative=self.s)
        return f

def make_A():
    Nb_test=N+1
    Nlb_test=2
    Nb_trial=N+1
    Nlb_trial =2
    Ab=np.zeros((Nb_test,Nb_trial))
    for i in range(N):
        for alpha  in range(Nlb_trial):
            for beta in range(Nlb_test):
                integrand_function=IntegrandFunction(Pb[i],Pb[i+1],1,1,alpha,beta)
                value= integral(Pb[i],Pb[i+1],4,integrand_function.A_F)
                Ab[Tb[beta,i],Tb[alpha,i]]+=value
    return Ab

def make_B():
    Nlb_test = 2
    Bb = np.zeros(N+1)
    for i in range(N):
        for beta in range(Nlb_test):
            integrand_function = IntegrandFunction(Pb[i], Pb[i + 1], 0, 0, None, beta)
            value = integral(Pb[i], Pb[i + 1], 4, integrand_function.B_F)
            Bb[Tb[beta, i]] += value
    return Bb

def generate_Gauss_point_w(gauss_point_number):
    w,t=[],[]
    if gauss_point_number==4:
        w=np.array([0.3478548451,0.3478548451,0.6521451549,0.6521451549])
        t=np.array([0.8611363116,-0.8611363116,0.3399810436,-0.3399810436])
    if gauss_point_number==8:
        w=np.array([0.1012285363,0.1012285363,0.2223810345,0.2223810345,0.3137066459,0.3137066459,0.3626837834,0.3626837834])
        t=np.array([0.9602898565,-0.9602898565,0.7966664774,-0.7966664774,0.5255324099,-0.5255324099,0.1834346425,-0.1834346425])
    if gauss_point_number==2:
        w=np.array([1,1])
        t=np.array([-1/np.sqrt(3),1/np.sqrt(3)])
    return w,t

def integral(a,b,gauss_point_number,f):
    w,t=generate_Gauss_point_w(gauss_point_number)
    w=((b-a)/2)*w
    x=(b+a)/2+((b-a)/2)*t
    result=0.0
    for i in range(gauss_point_number):
        result+=w[i]*f(x[i])
    return result

def boundary_treatment(A,B):
    for i in range(N+1):
        gx=G(Pb[i])
        if gx != 'No limit':
            A[i, :] = 0
            A[i, i] = 1
            B[i] = gx

def get_maximum_error(solution):
    maxerror=0
    for i in range(N+1):
        temp=solution[i]-U(Pb[i])
        if abs(maxerror)<abs(temp):
            maxerror=abs(temp)
    return maxerror

if __name__ == '__main__':
    for i in range(2,8):
        h = 1 / (2**i)
        left, right = x_range
        N = int((right - left) / h)

        Pb = make_Pb()
        Tb = make_Tb()

        A = make_A()
        B = make_B()
        boundary_treatment(A,B)
        solution = np.linalg.solve(A, B)
        print('h is 1/%s maxerror is %.4e '%(str(2**i),get_maximum_error(solution)))