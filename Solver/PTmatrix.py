import numpy as np
from enum import Enum


class PMatrixType(Enum):
    One_Dimensional_Linear_Node = 1

class TMatrixType(Enum):
    One_Dimensional_Linear_Cell = 1
    One_Dimensional_Quadratic_Cell = 2

class PTMatrix():
    def __init__(self,range,h_Mesh,h_Finite):
        self.range=range
        self.h_Mesh = h_Mesh
        self.h_Finite =h_Finite

        self.pMatrixType=PMatrixType.One_Dimensional_Linear_Node
        self.tMatrixType=TMatrixType.One_Dimensional_Linear_Cell
        self.pbMatrixType = PMatrixType.One_Dimensional_Linear_Node
        self.tbMatrixType = TMatrixType.One_Dimensional_Linear_Cell

    def _make_One_Dimensional_Linear_Node(self,left,right,h):
        Node_num = int((right -left)/h)+1
        P = np.zeros(Node_num)
        for i in range(Node_num):
            P[i] = left + i * h
        return P

    def _make_One_Dimensional_Linear_Cell(self,N):
        Tb = np.zeros((2, N), dtype=int)
        for i in range(N):
            Tb[0, i] = i
            Tb[1, i] = i + 1
        return Tb


    def _make_One_Dimensional_Quadratic_Cell(self,N):
        Tb = np.zeros((3, N), dtype=int)
        for i in range(N):
            Tb[0, i] = 2 * i
            Tb[1, i] = 2 * i + 2
            Tb[2, i] = 2 * i + 1
        return Tb

    def getPTMatrix(self):
        if self.pMatrixType == PMatrixType.One_Dimensional_Linear_Node:
            self.left, self.right = self.range
            self.N = int((self.right - self.left) / self.h_Mesh)
            self.Nm = self.N + 1
            self.P = self._make_One_Dimensional_Linear_Node(self.left, self.right, self.h_Mesh)

        if self.tMatrixType == TMatrixType.One_Dimensional_Linear_Cell:
            self.T=self._make_One_Dimensional_Linear_Cell(self.N)

        if self.tMatrixType == TMatrixType.One_Dimensional_Quadratic_Cell:
            self.T = self._make_One_Dimensional_Quadratic_Cell(self.N)

    def getPbTbMatrix(self):
        if self.pbMatrixType == PMatrixType.One_Dimensional_Linear_Node:
            left, right = self.range
            self.Nb = int((right - left) / self.h_Finite)
            self.Nbm = self.Nb + 1
            self.Pb = self._make_One_Dimensional_Linear_Node(left, right, self.h_Finite)

        if self.tbMatrixType == TMatrixType.One_Dimensional_Linear_Cell:
            self.Tb = self._make_One_Dimensional_Linear_Cell(self.N)

        if self.tbMatrixType == TMatrixType.One_Dimensional_Quadratic_Cell:
            self.Tb = self._make_One_Dimensional_Quadratic_Cell(self.N)
