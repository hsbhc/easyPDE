#-- coding: utf-8 --
'''
@Project: easyPDE
@File: solvers.py
@Author: SryM.HJ
@Ide: PyCharm
@Time: 2021/12/6 20:51
@Function: Solver
'''

from finiteElement2D_t.PTmatrix import PTMatrix, PMatrixType, TMatrixType
from finiteElement2D_t.basisFunction import BasisFunction, BasisFunctionType
from finiteElement2D_t.boundaryProcessor import BoundaryProcessor
from finiteElement2D_t.integrandFunctions import  getIntegrandFunction
from finiteElement2D_t.integrators import IntegratorType
from finiteElement2D_t.linearSystem import LinearSystem
from finiteElement2D_t.question import Question, One_Dimensional_PoissonQuestion
from finiteElement2D_t.integrators import getIntegrator
import numpy as np

class Solver_t():
    def __init__(self,question:Question,h_Mesh,h_Finite,h_t=0.1,scheme=0):
        self.question = question
        self.h_t=h_t
        self.scheme=scheme
        self.m=int((self.question.t_range[1]-self.question.t_range[0])/h_t)+1
        if isinstance(self.question, (One_Dimensional_PoissonQuestion)):
            self.h_Mesh = (h_Mesh,)
            self.h_Finite = (h_Finite,)
        else:
            self.h_Mesh = h_Mesh
            self.h_Finite = h_Finite

    def setBasisFunction(self, trialBasisFunctionType: BasisFunctionType, testBasisFunctionType: BasisFunctionType):
        '''
        set basisFunction and Nlb
        '''
        self.trialBasisFunctionType = trialBasisFunctionType
        self.testBasisFunctionType = testBasisFunctionType
        self.basisFunction = BasisFunction()
        self.Nlb_trial, self.trial, self.Nlb_test, self.test = \
            self.basisFunction.get_trial_test(self.trialBasisFunctionType,
                                              self.testBasisFunctionType)

    def setIntegrator(self, integratorType: IntegratorType):
        '''
        set integrator
        '''
        self.integratorType = integratorType
        self.integrator = getIntegrator(self.integratorType)

    def setPT_PbTb(self, pMatrixType: PMatrixType, tMatrixType: TMatrixType, pbMatrixType: PMatrixType,
                   tbMatrixType: TMatrixType):
        '''
        set PT and PbTb
        '''
        self.PT_matrix = PTMatrix(self.question.range, self.h_Mesh, self.h_Finite)
        self.PT_matrix.setPT_PbTb_type(pMatrixType, tMatrixType, pbMatrixType, tbMatrixType)
        self.PT_matrix.getPTMatrix()
        self.PT_matrix.getPbTbMatrix()
        self.PT_matrix.h_t=self.h_t
        self.PT_matrix.scheme=self.scheme
    def makeLinearSystem(self):
        '''
        Generate matrix equation
        '''
        integrandFunction = getIntegrandFunction(self.question, self.trial, self.test)
        self.linearSystem = LinearSystem(self.PT_matrix, self.Nlb_trial, self.Nlb_test, self.integrator,
                                         integrandFunction)
        matrixEquation=self.linearSystem.getMatrixEquation()
        A=matrixEquation.A
        M=matrixEquation.M
        if self.scheme != 0:
            A_ = M / (self.h_t * self.scheme) + A
        else:
            A_ = M / self.h_t

        X_old=self.getX0()
        self.X=[X_old]
        for i in range(1,self.m):
            t0 = self.question.t_range[0] + self.h_t * (i-1)
            t1=self.question.t_range[0]+self.h_t*i
            linearSystem = LinearSystem(self.PT_matrix,self.Nlb_trial,self.Nlb_test,self.integrator,integrandFunction)
            linearSystem.integrandFunction.t=t0
            B0 = linearSystem.getB()
            linearSystem.integrandFunction.t = t1
            B1 = linearSystem.getB()

            if self.scheme != 0:
                B_ = self.scheme * B1 + (1 - self.scheme) * B0 + np.dot(M, X_old) / (self.h_t * self.scheme)
            else:
                B_= self.scheme * B1 + (1 - self.scheme) * B0 + np.dot(A_-A, X_old)

            matrixEquation.A=A_
            matrixEquation.B=B_
            boundaryProcessor = BoundaryProcessor(self.question, self.PT_matrix, linearSystem)
            matrixEquation = boundaryProcessor.boundary_treatment(matrixEquation)
            X=np.linalg.solve(matrixEquation.A,matrixEquation.B)

            if self.scheme!=0:
                X = (X - X_old) / self.scheme + X_old
            X_old=X
            self.X.append(X_old)

    def getX0(self):
        X_old= np.zeros(self.PT_matrix.Nbm)
        for i in range(self.PT_matrix.Nbm):
            point =self.PT_matrix.Pb[:,i]
            X_old[i]=self.question.T0(point[0],point[1])
        return X_old