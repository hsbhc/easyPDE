# -- coding: utf-8 --
'''
@Project: easyPDE
@File: boundaryProcessor.py
@Author: SryM.HJ
@Ide: PyCharm
@Time: 2021/12/6 20:38
@Function: BoundaryProcessor
'''

from finiteElement2D_t.PTmatrix import PTMatrix, EN
from finiteElement2D_t.integrators import getIntegrator, IntegratorType
from finiteElement2D_t.linearSystem import MatrixEquation, LinearSystem
from finiteElement2D_t.question import Question
import numpy as np


class BoundaryProcessor():
    def __init__(self, question: Question, PT_matrix: PTMatrix, linearSystem: LinearSystem):
        self._question = question
        self._range = question.range
        self._PT_matrix = PT_matrix
        self._linearSystem = linearSystem

    def boundary_treatment(self, matrixEquation: MatrixEquation):
        if hasattr(self._question, 'dG'):
            matrixEquation = self._neumann_boundary_treatment(matrixEquation)
        if hasattr(self._question, 'dGQbPb'):
            matrixEquation = self._robin_boundary_treatment(matrixEquation)
        if hasattr(self._question, 'G'):
            matrixEquation = self._dirichlet_boundary_treatment(matrixEquation)

        return matrixEquation

    def _dirichlet_boundary_treatment(self, matrixEquation: MatrixEquation):
        '''
        Dirichlet boundary condition
        '''
        if self._PT_matrix.En.dim == 1:
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

        if self._PT_matrix.En.dim == 2:
            for i in range(self._PT_matrix.Nbm):
                x, y = self._PT_matrix.Pb[:, i]
                if x == self._range[0][0] or x == self._range[0][1] or y == self._range[1][0] or y == self._range[1][1]:
                    gx = self._question.G(x, y,self._linearSystem.integrandFunction.t)
                    gx0 = self._question.G(x, y, self._linearSystem.integrandFunction.t-self._PT_matrix.h_t)
                    if gx != 'No limit':
                        matrixEquation.A[i, :] = 0
                        matrixEquation.A[i, i] = 1
                        if self._PT_matrix.scheme==0:
                            matrixEquation.B[i] = gx
                        else:
                            matrixEquation.B[i] = self._PT_matrix.scheme * gx + (1 - self._PT_matrix.scheme) * gx0
            return matrixEquation

    def _neumann_boundary_treatment(self, matrixEquation: MatrixEquation):
        '''
        Neumann boundary condition
        '''
        if self._PT_matrix.En.dim == 1:
            for i in range(self._PT_matrix.Nbm):
                x = self._PT_matrix.Pb[i]
                if x not in self._range:
                    continue
                gx = self._question.dG(x)
                if gx != 'No limit':
                    if x == self._range[0]:
                        matrixEquation.B[i] = matrixEquation.B[i] - gx * self._question.C(x)
                    if x == self._range[1]:
                        matrixEquation.B[i] = matrixEquation.B[i] + gx * self._question.C(x)
            return matrixEquation

        if self._PT_matrix.En.dim == 2:
            integrator = getIntegrator(IntegratorType.Gauss)
            integrator.set_gauss_point_number(4)
            Vb = np.zeros(self._PT_matrix.Nbm)
            for i in range(self._PT_matrix.N):
                element = self._PT_matrix.T[:, i]
                for n1 in range(len(element)):
                    for n2 in range(n1 + 1, len(element)):
                        x1, y1 = self._PT_matrix.P[:, element[n1]]
                        x2, y2 = self._PT_matrix.P[:, element[n2]]
                        gx1, gx2 = self._question.dG(x1, y1), self._question.dG(x2, y2)
                        if gx1 != 'No limit' and gx2 != 'No limit':
                            en = EN([0], dim=1)
                            for beta in range(self._linearSystem.Nlb_test):
                                if x1 == x2:
                                    en.correct_En = np.array([[y1, ], [y2, ]]) if y2 > y1 else np.array(
                                        [[y2, ], [y1, ]])
                                    V_F = lambda y: self._question.dG(x1, y) * self._question.C(x1,
                                                                                                y) * self._linearSystem.integrandFunction.test(
                                        x1, y, self._PT_matrix.En.getEnByIndex(i).correct_En, beta, 0, 0)
                                    value = integrator.integral(en, V_F)
                                    Vb[self._PT_matrix.Tb[beta, i]] += value
                                if y1 == y2:
                                    en.correct_En = np.array([[x1, ], [x2, ]]) if x2 > x1 else np.array(
                                        [[x2, ], [x1, ]])
                                    V_F = lambda x: self._question.dG(x, y1) * self._question.C(x,
                                                                                                y1) * self._linearSystem.integrandFunction.test(
                                        x, y1, self._PT_matrix.En.getEnByIndex(i).correct_En, beta, 0, 0)
                                    value = integrator.integral(en, V_F)
                                    Vb[self._PT_matrix.Tb[beta, i]] += value

            matrixEquation.B = matrixEquation.B + Vb
            return matrixEquation

    def _robin_boundary_treatment(self, matrixEquation: MatrixEquation):
        '''
        Robin boundary condition
        '''
        if self._PT_matrix.En.dim == 1:
            for i in range(self._PT_matrix.Nbm):
                x = self._PT_matrix.Pb[i]
                if x not in self._range:
                    continue
                gx = self._question.dGQbPb(x)
                if gx != 'No limit':
                    Q, P = gx
                    if x == self._range[0]:
                        matrixEquation.A[i][i] = matrixEquation.A[i][i] - Q * self._question.C(x)
                        matrixEquation.B[i] = matrixEquation.B[i] - P * self._question.C(x)
                    if x == self._range[1]:
                        matrixEquation.A[i][i] = matrixEquation.A[i][i] + Q * self._question.C(x)
                        matrixEquation.B[i] = matrixEquation.B[i] + P * self._question.C(x)
            return matrixEquation

        if self._PT_matrix.En.dim == 2:
            integrator = getIntegrator(IntegratorType.Gauss)
            integrator.set_gauss_point_number(4)
            Rb = np.zeros((self._PT_matrix.Nbm, self._PT_matrix.Nbm))
            Wb = np.zeros(self._PT_matrix.Nbm)
            for i in range(self._PT_matrix.N):
                element = self._PT_matrix.T[:, i]
                for n1 in range(len(element)):
                    for n2 in range(n1 + 1, len(element)):
                        x1, y1 = self._PT_matrix.P[:, element[n1]]
                        x2, y2 = self._PT_matrix.P[:, element[n2]]
                        gx1 = self._question.dGQbPb(x1, y1)
                        gx2 = self._question.dGQbPb(x2, y2)
                        if gx1 != 'No limit' and gx2 != 'No limit':
                            r, q = gx1
                            en = EN([0], dim=1)

                            for beta1 in range(self._linearSystem.Nlb_test):
                                if x1 == x2:
                                    en.correct_En = np.array([[y1, ], [y2, ]]) if y2 > y1 else np.array(
                                        [[y2, ], [y1, ]])
                                    V_F = lambda y: q * self._question.C(x1,
                                                                         y) * self._linearSystem.integrandFunction.test(
                                        x1, y, self._PT_matrix.En.getEnByIndex(i).correct_En, beta1, 0, 0)
                                    value = integrator.integral(en, V_F)
                                    Wb[self._PT_matrix.Tb[beta1, i]] += value
                                if y1 == y2:
                                    en.correct_En = np.array([[x1, ], [x2, ]]) if x2 > x1 else np.array(
                                        [[x2, ], [x1, ]])
                                    V_F = lambda x: q * self._question.C(x,
                                                                         y1) * self._linearSystem.integrandFunction.test(
                                        x, y1, self._PT_matrix.En.getEnByIndex(i).correct_En, beta1, 0, 0)
                                    value = integrator.integral(en, V_F)
                                    Wb[self._PT_matrix.Tb[beta1, i]] += value

                            for alpha in range(self._linearSystem.Nlb_trial):
                                for beta in range(self._linearSystem.Nlb_test):
                                    if x1 == x2:
                                        en.correct_En = np.array([[y1, ], [y2, ]]) if y2 > y1 else np.array(
                                            [[y2, ], [y1, ]])
                                        R_F = lambda y: r * self._question.C(x1,
                                                                             y) * self._linearSystem.integrandFunction.trial(
                                            x1, y, self._PT_matrix.En.getEnByIndex(i).correct_En, alpha, 0, 0) * \
                                                        self._linearSystem.integrandFunction.test(x1, y,
                                                                                                  self._PT_matrix.En.getEnByIndex(
                                                                                                      i).correct_En,
                                                                                                  beta, 0, 0)
                                        value = integrator.integral(en, R_F)
                                        Rb[self._PT_matrix.Tb[beta, i], self._PT_matrix.Tb[alpha, i]] += value
                                    if y1 == y2:
                                        en.correct_En = np.array([[x1, ], [x2, ]]) if x2 > x1 else np.array(
                                            [[x2, ], [x1, ]])
                                        R_F = lambda x: r * self._question.C(x,
                                                                             y1) * self._linearSystem.integrandFunction.trial(
                                            x, y1, self._PT_matrix.En.getEnByIndex(i).correct_En, alpha, 0, 0) * \
                                                        self._linearSystem.integrandFunction.test(x, y1,
                                                                                                  self._PT_matrix.En.getEnByIndex(
                                                                                                      i).correct_En,
                                                                                                  beta, 0, 0)
                                        value = integrator.integral(en, R_F)
                                        Rb[self._PT_matrix.Tb[beta, i], self._PT_matrix.Tb[alpha, i]] += value

            matrixEquation.A = matrixEquation.A + Rb
            matrixEquation.B = matrixEquation.B + Wb
            return matrixEquation
