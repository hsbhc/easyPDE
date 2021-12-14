#-- coding: utf-8 --
'''
@Project: easyPDE
@File: solvers.py
@Author: SryM.HJ
@Ide: PyCharm
@Time: 2021/12/6 20:51
@Function: Solver
'''

from finiteElement2D.PTmatrix import PTMatrix, PMatrixType, TMatrixType
from finiteElement2D.basisFunction import BasisFunction, BasisFunctionType
from finiteElement2D.boundaryProcessor import BoundaryProcessor
from finiteElement2D.integrandFunctions import  getIntegrandFunction
from finiteElement2D.integrators import IntegratorType
from finiteElement2D.linearSystem import LinearSystem
from finiteElement2D.question import Question, One_Dimensional_PoissonQuestion
from finiteElement2D.integrators import getIntegrator

class Solver():
    def __init__(self,question:Question,h_Mesh,h_Finite):
        self.question = question
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

    def makeLinearSystem(self):
        '''
        Generate matrix equation
        '''
        self.integrandFunction = getIntegrandFunction(self.question, self.trial, self.test)
        self.linearSystem = LinearSystem(self.PT_matrix,self.Nlb_trial,self.Nlb_test,self.integrator,self.integrandFunction)
        self.matrixEquation = self.linearSystem.getMatrixEquation()
    def makeBoundaryProcess(self):
        '''
        BoundaryProcess
        '''
        self.boundaryProcessor = BoundaryProcessor(self.question,self.PT_matrix,self.linearSystem)
        self.matrixEquation = self.boundaryProcessor.boundary_treatment(self.matrixEquation)

    def solve(self):
        self.matrixEquation.solve()


    def U(self,x):
        '''
        Numerical solution
        '''
        for i in range(self.PT_matrix.N):
            a, b = self.PT_matrix.P[i], self.PT_matrix.P[i + 1]
            if x>=a and x<=b:
                numericalSolution = 0
                for k in range(self.Nlb_trial):
                    numericalSolution += self.matrixEquation.solution[self.PT_matrix.Tb[k, i]] * self.trial(x,(a,b),k,derivative=0)
                return numericalSolution

        return None