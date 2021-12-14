#-- coding: utf-8 --
'''
@Project: easyPDE
@File: PTmatrix.py
@Author: SryM.HJ
@Ide: PyCharm
@Time: 2021/12/6 20:05
@Function: The generation rules of P,T,Pb,Tb are defined, PTMatrix also stores N, Nm, Nb, Nbm ( N---Nb, Nm---Nbm)
'''

import numpy as np
from enum import Enum

class PMatrixType(Enum):
    One_Dimensional_Linear_Node = 1

class TMatrixType(Enum):
    One_Dimensional_Linear_Cell = 1
    One_Dimensional_Quadratic_Cell = 2

class PTMatrix():
    def __init__(self,range,h_Mesh,h_Finite):
        '''
        Initialize a PTMatrix
        :param range: Variable range such as (0,1)
        :param h_Mesh: Division step of mesh element
        :param h_Finite: Division step of finite element
        '''
        self._range=range
        self._h_Mesh = h_Mesh
        self._h_Finite =h_Finite

        self._pMatrixType=PMatrixType.One_Dimensional_Linear_Node
        self._tMatrixType=TMatrixType.One_Dimensional_Linear_Cell
        self._pbMatrixType = PMatrixType.One_Dimensional_Linear_Node
        self._tbMatrixType = TMatrixType.One_Dimensional_Linear_Cell

    def setPT_PbTb_type(self,pMatrixType:PMatrixType,tMatrixType:TMatrixType,pbMatrixType:PMatrixType,tbMatrixType:TMatrixType):
        self._pMatrixType = pMatrixType
        self._tMatrixType = tMatrixType
        self._pbMatrixType = pbMatrixType
        self._tbMatrixType = tbMatrixType

    def _make_One_Dimensional_Linear_Node(self,left,right,h):
        '''
        Generate one-dimensional linear nodes according to the range and step size
        Corresponding to P and Pb
        :param left: Left boundary such as 0
        :param right: Right boundary such as 1
        :param h: Step size such as 0.5
        :return: P: [0,0.5,1]
        '''
        Node_num = int((right -left)/h)+1
        P = np.zeros(Node_num)
        for i in range(Node_num):
            P[i] = left + i * h
        return P

    def _make_One_Dimensional_Linear_Cell(self,N):
        '''
        Global index of Linear element nodes in mesh element according to the number of Mesh elements
        Corresponding to T and Tb
        :param N: Number of Mesh elements such as 2
        :return: Tb: [[0,1],
                      [1,2]]
        '''
        Tb = np.zeros((2, N), dtype=int)
        for i in range(N):
            Tb[0, i] = i
            Tb[1, i] = i + 1
        return Tb


    def _make_One_Dimensional_Quadratic_Cell(self,N):
        '''
        Global index of Quadratic element nodes in mesh element according to the number of Mesh elements
        Corresponding to T and Tb
        :param N: Number of Mesh elements such as 2
        :return: Tb: [[0,2],
                       [2,4],
                       [1,3]]
        '''
        Tb = np.zeros((3, N), dtype=int)
        for i in range(N):
            Tb[0, i] = 2 * i
            Tb[1, i] = 2 * i + 2
            Tb[2, i] = 2 * i + 1
        return Tb

    def getPTMatrix(self):
        '''
        Get P, T, N, Nm by self.pMatrixType,self.tMatrixType
        '''
        if self._pMatrixType == PMatrixType.One_Dimensional_Linear_Node:
            self.left, self.right = self._range
            self.N = int((self.right - self.left) / self._h_Mesh)
            self.Nm = self.N + 1
            self.P = self._make_One_Dimensional_Linear_Node(self.left, self.right, self._h_Mesh)

        if self._tMatrixType == TMatrixType.One_Dimensional_Linear_Cell:
            self.T=self._make_One_Dimensional_Linear_Cell(self.N)

        if self._tMatrixType == TMatrixType.One_Dimensional_Quadratic_Cell:
            self.T = self._make_One_Dimensional_Quadratic_Cell(self.N)

    def getPbTbMatrix(self):
        '''
        Get Pb, Tb, Nb, Nbm by self.pMatrixType,self.tMatrixType
        '''
        if self._pbMatrixType == PMatrixType.One_Dimensional_Linear_Node:
            left, right = self._range
            self.Nb = int((right - left) / self._h_Finite)
            self.Nbm = self.Nb + 1
            self.Pb = self._make_One_Dimensional_Linear_Node(left, right, self._h_Finite)

        if self._tbMatrixType == TMatrixType.One_Dimensional_Linear_Cell:
            self.Tb = self._make_One_Dimensional_Linear_Cell(self.N)

        if self._tbMatrixType == TMatrixType.One_Dimensional_Quadratic_Cell:
            self.Tb = self._make_One_Dimensional_Quadratic_Cell(self.N)
