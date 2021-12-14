import numpy as np
from Solver.PTmatrix import PTMatrix
from Solver.integrators import Integrator


class MatrixEquation():
    def __init__(self,A,B):
        self.A=A
        self.B=B
    def solve(self):
        self.solution = np.linalg.solve(self.A, self.B)

class LinearSystem():
    def __init__(self,PT_matrix:PTMatrix,Nlb_trial,Nlb_test,integrator:Integrator,integrandFunction):
        self.PT_matrix = PT_matrix
        self.Nlb_trial=Nlb_trial
        self.Nlb_test=Nlb_test

        self.integrandFunction=integrandFunction
        self.integrator=integrator


    def _make_A(self):
        Ab=np.zeros((self.PT_matrix.Nbm,self.PT_matrix.Nbm))
        for i in range(self.PT_matrix.N):
            for alpha  in range(self.Nlb_trial):
                for beta in range(self.Nlb_test):
                    self.integrandFunction.set(self.PT_matrix.P[i],self.PT_matrix.P[i+1],alpha,beta)
                    value= self.integrator.integral(self.PT_matrix.P[i],self.PT_matrix.P[i+1],self.integrandFunction.A_F)
                    Ab[self.PT_matrix.Tb[beta,i],self.PT_matrix.Tb[alpha,i]]+=value
        return Ab

    def _make_B(self):
        Bb = np.zeros(self.PT_matrix.Nbm)
        for i in range(self.PT_matrix.N):
            for beta in range(self.Nlb_test):
                self.integrandFunction.set(self.PT_matrix.P[i], self.PT_matrix.P[i + 1],None, beta)
                value = self.integrator.integral(self.PT_matrix.P[i], self.PT_matrix.P[i + 1],self.integrandFunction.B_F)
                Bb[self.PT_matrix.Tb[beta, i]] += value
        return Bb

    def getMatrixEquation(self):
        return MatrixEquation(self._make_A(),self._make_B())