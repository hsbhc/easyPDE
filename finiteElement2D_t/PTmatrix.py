#-- coding: utf-8 --
'''
@Project: easyPDE
@File: PTmatrix.py
@Author: SryM.HJ
@Ide: PyCharm
@Time: 2021/12/6 20:05
@Function: The generation rules of P,T,Pb,Tb are defined, PTMatrix also stores N, Nm, Nb, Nbm ( N---Nb, Nm---Nbm)  En
'''

import numpy as np
from enum import Enum

class EN():
    def __init__(self,elements,dim):
        self.dim=dim
        self.elements=elements
        self.correct_En=elements[0]
    def getEnByIndex(self,i):
        self.correct_En=self.elements[i]
        return self

class PMatrixType(Enum):
    One_Dimensional_Linear_Node = 1
    Two_Dimensional_Linear_Node=2

class TMatrixType(Enum):
    One_Dimensional_Linear_Cell = 1
    One_Dimensional_Quadratic_Cell = 2
    Two_Dimensional_Linear_Cell=3
    Two_Dimensional_Quadratic_Cell=4

class PTMatrix():
    def __init__(self,range,h_Mesh,h_Finite):
        '''
        Initialize a PTMatrix
        :param range: Variable range such as (0,1)
        :param h_Mesh: Division step of mesh element such as (0.5,)
        :param h_Finite: Division step of finite element such as (0.5,)
        '''
        self._range=range
        self._h_Mesh = h_Mesh
        self._h_Finite =h_Finite
        self.h_t=0
        self.scheme=0
        self._pMatrixType=PMatrixType.One_Dimensional_Linear_Node
        self._tMatrixType=TMatrixType.One_Dimensional_Linear_Cell
        self._pbMatrixType = PMatrixType.One_Dimensional_Linear_Node
        self._tbMatrixType = TMatrixType.One_Dimensional_Linear_Cell

    def setPT_PbTb_type(self,pMatrixType:PMatrixType,tMatrixType:TMatrixType,pbMatrixType:PMatrixType,tbMatrixType:TMatrixType):
        self._pMatrixType = pMatrixType
        self._tMatrixType = tMatrixType
        self._pbMatrixType = pbMatrixType
        self._tbMatrixType = tbMatrixType

    def _make_One_Dimensional_Linear_Node(self,ranges,h):
        '''
        Generate one-dimensional linear nodes according to the range and step size
        Corresponding to P and Pb
        :param ranges: range such as (0,1)
        :param h: Step size such as (0.5,)
        :return: P: [0,0.5,1]
        '''
        Node_num = int((ranges[1] -ranges[0])/h[0])+1
        P = np.zeros(Node_num)
        for i in range(Node_num):
            P[i] = ranges[0] + i * h[0]
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

    def _make_Two_Dimensional_Linear_Node(self,ranges, h):
        '''
        Generate Two-dimensional linear nodes according to the range and step size
        Corresponding to P and Pb
        :param ranges: range such as ((0,1),(0,1))
        :param h: Step size such as (0.5,0.5)
        :return: P: [[0.  0.  0.  0.5 0.5 0.5 1.  1.  1. ]
                    [0.  0.5 1.  0.  0.5 1.  0.  0.5 1. ]]
        '''
        N1 = int((ranges[0][1] - ranges[0][0]) / h[0])
        N2 = int((ranges[1][1] - ranges[1][0]) / h[1])
        N1_m = N1 + 1
        N2_m = N2 + 1
        Node_num = N1_m * N2_m
        P = np.zeros((2, Node_num))
        for rn in range(N1_m):
            for cn in range(N2_m):
                x = ranges[0][0] + rn * h[0]
                y = ranges[1][0] + cn * h[1]

                j = rn * N2_m + cn
                P[0, j] = x
                P[1, j] = y
        return P

    def _make_Two_Dimensional_Linear_Cell(self,N1, N2):
        '''
        Global index of Two dimensional linear element nodes in mesh element according to the number of Mesh elements
        Corresponding to T and Tb
        :param N: Number of Mesh elements such as 2 ,2
        :return: Tb: [[0 1 1 2 3 4 4 5]
                     [3 3 4 4 6 6 7 7]
                    [1 4 2 5 4 7 5 8]]
        '''
        N = 2 * N1 * N2
        Tb = np.zeros((3, N), dtype=int)

        for ce in range(N1):
            for re in range(N2):
                n1 = ce * 2 * N2 + 2 * re
                n2 = ce * 2 * N2 + 2 * re + 1

                j1 = re + ce * (N2 + 1)
                j2 = re + ce * (N2 + 1) + 1
                j3 = re + (ce + 1) * (N2 + 1)
                j4 = re + (ce + 1) * (N2 + 1) + 1

                Tb[0, n1] = j1
                Tb[1, n1] = j3
                Tb[2, n1] = j2
                Tb[0, n2] = j2
                Tb[1, n2] = j3
                Tb[2, n2] = j4

        return Tb


    def _make_Two_Dimensional_Quadratic_Cell(self,N1, N2):
        '''
        Global index of Two_Dimensional Quadratic element nodes in mesh element according to the number of Mesh elements
        Corresponding to T and Tb
        :param N: Number of Mesh elements such as 2 , 2
        :return: Tb: [[ 0  2  2  4 10 12 12 14]
                        [10 10 12 12 20 20 22 22]
                        [ 2 12  4 14 12 22 14 24]
                        [ 5  6  7  8 15 16 17 18]
                        [ 6 11  8 13 16 21 18 23]
                        [ 1  7  3  9 11 17 13 19]]
        '''
        N = 2 * N1 * N2

        Tb = np.zeros((6, N), dtype=int)
        for ce in range(N1):
            for re in range(N2):
                n1 = ce * 2 * N2 + 2 * re
                n2 = ce * 2 * N2 + 2 * re + 1

                center = (2 * N2 + 2) + 2 * (2 * N2 + 1) * ce + 2 * re
                j1 = center - (2 * N2 + 1) - 1
                j2 = center - (2 * N2 + 1)
                j3 = center - (2 * N2 + 1) + 1
                j4 = center - 1
                j5 = center
                j6 = center + 1
                j7 = center + (2 * N2 + 1) - 1
                j8 = center + (2 * N2 + 1)
                j9 = center + (2 * N2 + 1) + 1
                Tb[0, n1] = j1
                Tb[1, n1] = j7
                Tb[2, n1] = j3
                Tb[3, n1] = j4
                Tb[4, n1] = j5
                Tb[5, n1] = j2

                Tb[0, n2] = j3
                Tb[1, n2] = j7
                Tb[2, n2] = j9
                Tb[3, n2] = j5
                Tb[4, n2] = j8
                Tb[5, n2] = j6

        return Tb

    def _getEn(self):
        En=[]
        cell=[]
        if self.P.ndim == 1:
            for index in range(self.N):
                for p_index in self.T[:,index]:
                    cell.append([self.P[p_index],])
                En.append(cell)
                cell=[]
        if self.P.ndim == 2:
            for index in range(self.N):
                for p_index in self.T[:,index]:
                    cell.append(self.P[:,p_index])
                En.append(cell)
                cell=[]
        self.En=EN(np.array(En),dim=self.P.ndim)

    def getPTMatrix(self):
        '''
        Get P, T, N, Nm by self.pMatrixType,self.tMatrixType
        '''
        if self._pMatrixType == PMatrixType.One_Dimensional_Linear_Node:
            self.left, self.right = self._range
            self.N = int((self.right - self.left) / self._h_Mesh[0])
            self.Nm = self.N + 1
            self.P = self._make_One_Dimensional_Linear_Node(self._range,self._h_Mesh)
        if self._pMatrixType == PMatrixType.Two_Dimensional_Linear_Node:
            N1 = int((self._range[0][1] - self._range[0][0]) / self._h_Mesh[0])
            N2 = int((self._range[1][1] - self._range[1][0]) / self._h_Mesh[1])
            self.N=2*N1*N2
            self.Nm = (N1+1)*(N2+1)
            self.P = self._make_Two_Dimensional_Linear_Node(self._range, self._h_Mesh)

        if self._tMatrixType == TMatrixType.One_Dimensional_Linear_Cell:
            self.T=self._make_One_Dimensional_Linear_Cell(self.N)

        if self._tMatrixType == TMatrixType.One_Dimensional_Quadratic_Cell:
            self.T = self._make_One_Dimensional_Quadratic_Cell(self.N)

        if self._tMatrixType == TMatrixType.Two_Dimensional_Linear_Cell:
            N1 = int((self._range[0][1] - self._range[0][0]) / self._h_Mesh[0])
            N2 = int((self._range[1][1] - self._range[1][0]) / self._h_Mesh[1])
            self.T=self._make_Two_Dimensional_Linear_Cell(N1,N2)

        if self._tMatrixType == TMatrixType.Two_Dimensional_Quadratic_Cell:
            N1 = int((self._range[0][1] - self._range[0][0]) / self._h_Mesh[0])
            N2 = int((self._range[1][1] - self._range[1][0]) / self._h_Mesh[1])
            self.T = self._make_Two_Dimensional_Quadratic_Cell(N1,N2)


        self._getEn()
    def getPbTbMatrix(self):
        '''
        Get Pb, Tb, Nb, Nbm by self.pMatrixType,self.tMatrixType
        '''
        if self._pbMatrixType == PMatrixType.One_Dimensional_Linear_Node:
            left, right = self._range
            self.Nb = int((right - left) / self._h_Finite[0])
            self.Nbm = self.Nb + 1
            self.Pb = self._make_One_Dimensional_Linear_Node(self._range, self._h_Finite)
        if self._pbMatrixType == PMatrixType.Two_Dimensional_Linear_Node:
            N1 = int((self._range[0][1] - self._range[0][0]) / self._h_Finite[0])
            N2 = int((self._range[1][1] - self._range[1][0]) / self._h_Finite[1])
            self.Nb=2*N1*N2
            self.Nbm = (N1+1)*(N2+1)
            self.Pb = self._make_Two_Dimensional_Linear_Node(self._range, self._h_Finite)

        if self._tbMatrixType == TMatrixType.One_Dimensional_Linear_Cell:
            self.Tb = self._make_One_Dimensional_Linear_Cell(self.N)

        if self._tbMatrixType == TMatrixType.One_Dimensional_Quadratic_Cell:
            self.Tb = self._make_One_Dimensional_Quadratic_Cell(self.N)


        if self._tbMatrixType == TMatrixType.Two_Dimensional_Linear_Cell:
            N1 = int((self._range[0][1] - self._range[0][0]) / self._h_Mesh[0])
            N2 = int((self._range[1][1] - self._range[1][0]) / self._h_Mesh[1])
            self.Tb=self._make_Two_Dimensional_Linear_Cell(N1,N2)

        if self._tbMatrixType == TMatrixType.Two_Dimensional_Quadratic_Cell:
            N1 = int((self._range[0][1] - self._range[0][0]) / self._h_Mesh[0])
            N2 = int((self._range[1][1] - self._range[1][0]) / self._h_Mesh[1])
            self.Tb= self._make_Two_Dimensional_Quadratic_Cell(N1,N2)


        self._getEn()