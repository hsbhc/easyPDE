import numpy as np
from Solver.PTmatrix import PTMatrix, PMatrixType, TMatrixType
from Solver.basisFunction import BasisFunction, BasisFunctionType
from Solver.boundaryProcessor import BoundaryProcessor
from Solver.integrandFunctions import IntegrandFunction
from Solver.integrators import IntegratorType
from Solver.linearSystem import LinearSystem
from Solver.question import Question
from Solver.integrators import getIntegrator

np.set_printoptions(linewidth=np.inf)


class Solver():
    def __init__(self,question:Question,h_Mesh,h_Finite):
        self.question = question
        self.h_Mesh = h_Mesh
        self.h_Finite = h_Finite

    def makeLinearSystem(self):
        self.integrandFunction=IntegrandFunction(self.question,self.trial,self.test)
        self.linearSystem = LinearSystem(self.PT_matrix,self.Nlb_trial,self.Nlb_test,self.integrator,self.integrandFunction)
        self.matrixEquation = self.linearSystem.getMatrixEquation()

    def makeBoundaryProcess(self):
        self.boundaryProcessor = BoundaryProcessor(self.question,self.PT_matrix)
        self.matrixEquation = self.boundaryProcessor.boundary_treatment(self.matrixEquation)

    def solve(self):
        self.matrixEquation.solve()


    def setBasisFunctionType(self,trialBasisFunctionType:BasisFunctionType,testBasisFunctionType:BasisFunctionType):
        self.trialBasisFunctionType=trialBasisFunctionType
        self.testBasisFunctionType=testBasisFunctionType
        self.basisFunction = BasisFunction()
        self.Nlb_trial, self.trial, self.Nlb_test, self.test = \
            self.basisFunction.get_trial_test(self.trialBasisFunctionType,
                                              self.testBasisFunctionType)

    def setIntegratorType(self,integratorType:IntegratorType):
        self.integratorType=integratorType
        self.integrator=getIntegrator(self.integratorType)

    def setPT_PbTb(self,pMatrixType:PMatrixType,tMatrixType:TMatrixType,pbMatrixType:PMatrixType,tbMatrixType:TMatrixType):
        self.PT_matrix = PTMatrix(self.question.range,self.h_Mesh,self.h_Finite)
        self.PT_matrix.pMatrixType=pMatrixType
        self.PT_matrix.tMatrixType=tMatrixType
        self.PT_matrix.pbMatrixType=pbMatrixType
        self.PT_matrix.tbMatrixType=tbMatrixType
        self.PT_matrix.getPTMatrix()
        self.PT_matrix.getPbTbMatrix()


    def U(self,x):
        for i in range(self.PT_matrix.N):
            a, b = self.PT_matrix.P[i], self.PT_matrix.P[i + 1]
            if x>=a and x<=b:
                numericalSolution = 0
                for k in range(self.Nlb_trial):
                    numericalSolution += self.matrixEquation.solution[self.PT_matrix.Tb[k, i]] * self.trial(x,(a,b),k,derivative=0)
                return numericalSolution

        return None
