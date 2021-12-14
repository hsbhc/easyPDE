from Solver.PTmatrix import PTMatrix
from Solver.linearSystem import MatrixEquation
from Solver.question import Question

class BoundaryProcessor():
    def __init__(self,question:Question,PT_matrix:PTMatrix):
        self.question=question
        self.range=question.range
        self.PT_matrix = PT_matrix

    def boundary_treatment(self,matrixEquation:MatrixEquation):
        '''
        :param matrixEquation: MatrixEquation
        :return: Processed matrixEquation
        '''
        if hasattr(Question, 'G'):
            matrixEquation = self.dirichlet_boundary_treatment(matrixEquation)

        if hasattr(Question, 'dG'):
            matrixEquation = self.neumann_boundary_treatment(matrixEquation)

        if hasattr(Question, 'dGQbPb'):
            matrixEquation = self.robin_boundary_treatment(matrixEquation)

        return matrixEquation

    def dirichlet_boundary_treatment(self, matrixEquation: MatrixEquation):
        for i in range(self.PT_matrix.Nbm):
            x = self.PT_matrix.Pb[i]
            if x not in self.range:
                continue
            gx = self.question.G(x)
            if gx != 'No limit':
                matrixEquation.A[i, :] = 0
                matrixEquation.A[i, i] = 1
                matrixEquation.B[i] = gx
        return matrixEquation

    def neumann_boundary_treatment(self,matrixEquation:MatrixEquation):
        for i in range(self.PT_matrix.Nbm):
            x=self.PT_matrix.Pb[i]
            if x not in self.range:
                continue
            gx=self.question.dG(x)
            if gx != 'No limit':
                if x==self.range[0]:
                    matrixEquation.B[i] = matrixEquation.B[i]-gx*self.question.C(x)
                if x==self.range[1]:
                    matrixEquation.B[i] = matrixEquation.B[i] + gx * self.question.C(x)
        return matrixEquation

    def robin_boundary_treatment(self,matrixEquation:MatrixEquation):
        for i in range(self.PT_matrix.Nbm):
            x=self.PT_matrix.Pb[i]
            if x not in self.range:
                continue
            gx=self.question.dGQbPb(x)
            if gx != 'No limit':
                Q, P = gx
                if x==self.range[0]:
                    matrixEquation.A[i][i] = matrixEquation.A[i][i]-Q*self.question.C(x)
                    matrixEquation.B[i] = matrixEquation.B[i] - P * self.question.C(x)
                if x==self.range[1]:
                    matrixEquation.A[i][i] = matrixEquation.A[i][i] + Q * self.question.C(x)
                    matrixEquation.B[i] = matrixEquation.B[i] + P * self.question.C(x)
        return matrixEquation