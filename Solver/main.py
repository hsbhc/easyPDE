from Solver.PTmatrix import PMatrixType, TMatrixType
from Solver.basisFunction import BasisFunctionType
from Solver.evaluater import Evaluate
from Solver.integrators import IntegratorType
from Solver.question import Question
from Solver.solvers import Solver

if __name__ == '__main__':
    for i in range(2, 8):
        h = 1 / (2 ** i)
        question = Question()

        solver = Solver(question=question,h_Mesh=h,h_Finite=h)
        solver.setBasisFunctionType(trialBasisFunctionType=BasisFunctionType.One_Dimensional_Linear_Element,
                                    testBasisFunctionType=BasisFunctionType.One_Dimensional_Linear_Element)
        solver.setIntegratorType(IntegratorType.Gauss)
        solver.integrator.set_gauss_point_number(4)
        solver.setPT_PbTb(PMatrixType.One_Dimensional_Linear_Node, TMatrixType.One_Dimensional_Linear_Cell,
                      PMatrixType.One_Dimensional_Linear_Node, TMatrixType.One_Dimensional_Linear_Cell)


        solver.makeLinearSystem()
        solver.makeBoundaryProcess()
        solver.solve()

        evaluate = Evaluate(solver)
        print('h is 1/%s maxerror is %.4e ' % (str(2 ** i), evaluate.get_maximum_error()))
        # print('h is 1/%s Loo_norm error is %.4e ' % (str(2 ** i), evaluate.Loo_norm()))
        # print('h is 1/%s L2_norm error is %.4e ' % (str(2 ** i), evaluate.L2_norm()))
        # print('h is 1/%s H1_semi_norm error is %.4e ' % (str(2 ** i), evaluate.H1_semi_norm()))
