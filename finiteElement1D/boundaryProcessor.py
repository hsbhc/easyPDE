#-- coding: utf-8 --
'''
@Project: easyPDE
@File: boundaryProcessor.py
@Author: SryM.HJ
@Ide: PyCharm
@Time: 2021/12/6 20:38
@Function: BoundaryProcessor
'''

from finiteElement1D.PTmatrix import PTMatrix
from finiteElement1D.linearSystem import MatrixEquation
from finiteElement1D.question import Question

class BoundaryProcessor():
    def __init__(self,question:Question,PT_matrix:PTMatrix):
        self._question=question
        self._range=question.range
        self._PT_matrix = PT_matrix

    def boundary_treatment(self,matrixEquation:MatrixEquation):
        if hasattr(self._question, 'G'):
            matrixEquation = self._dirichlet_boundary_treatment(matrixEquation)

        if hasattr(self._question, 'dG'):
            matrixEquation = self._neumann_boundary_treatment(matrixEquation)

        if hasattr(self._question, 'dGQbPb'):
            matrixEquation = self._robin_boundary_treatment(matrixEquation)
        return matrixEquation

    def _dirichlet_boundary_treatment(self, matrixEquation: MatrixEquation):
        '''
        Dirichlet boundary condition
        '''
        for i in range(self._PT_matrix.Nbm):
            x = self._PT_matrix.Pb[i]
            if x not in self._range:
                continue
            gx = self._question.G(x)
            if gx != 'No limit':
                matrixEquation.A[i, :] = 0
                matrixEquation.A[i, i] = 1
                matrixEquation.B[i] = gx
        return matrixEquation

    def _neumann_boundary_treatment(self,matrixEquation:MatrixEquation):
        '''
        Neumann boundary condition
        '''
        for i in range(self._PT_matrix.Nbm):
            x=self._PT_matrix.Pb[i]
            if x not in self._range:
                continue
            gx=self._question.dG(x)
            if gx != 'No limit':
                if x==self._range[0]:
                    matrixEquation.B[i] = matrixEquation.B[i]-gx*self._question.C(x)
                if x==self._range[1]:
                    matrixEquation.B[i] = matrixEquation.B[i] + gx * self._question.C(x)
        return matrixEquation

    def _robin_boundary_treatment(self,matrixEquation:MatrixEquation):
        '''
        Robin boundary condition
        '''
        for i in range(self._PT_matrix.Nbm):
            x=self._PT_matrix.Pb[i]
            if x not in self._range:
                continue
            gx=self._question.dGQbPb(x)
            if gx != 'No limit':
                Q, P = gx
                if x==self._range[0]:
                    matrixEquation.A[i][i] = matrixEquation.A[i][i]-Q*self._question.C(x)
                    matrixEquation.B[i] = matrixEquation.B[i] - P * self._question.C(x)
                if x==self._range[1]:
                    matrixEquation.A[i][i] = matrixEquation.A[i][i] + Q * self._question.C(x)
                    matrixEquation.B[i] = matrixEquation.B[i] + P * self._question.C(x)
        return matrixEquation