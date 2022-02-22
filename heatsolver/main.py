from finiteElement3D_t.PTmatrix import PMatrixType, TMatrixType
from finiteElement3D_t.basisFunction import BasisFunctionType
from finiteElement3D_t.evaluater import Evaluate_t
from finiteElement3D_t.integrators import IntegratorType
from finiteElement3D_t.main import Question_Dirichlet_3D_t
from finiteElement3D_t.solvers import Solver_t
import numpy as np

from viewer.Data2VTK import FileVTK
from viewer.viewUtil import Viewer

if __name__ == '__main__':

    # 几何
    geometry = 'Cube'
    x_range = (0, 1)
    y_range = (0, 1)
    z_range = (0, 1)
    viewer = Viewer(x=x_range, y=y_range, z=z_range)
    viewer.plot_cube()

    # 网格
    grid = 'Cube'
    h_mesh = 0.125
    viewer.plot_cube_mesh(h_mesh=h_mesh, show_node=False)

    # 材料
    material = 'xxx'
    k = 100  # w/(m * K) = J/(s * m * K)
    rol = 10  # kg/m^3
    c = 10  # J/(kg * K)
    a_square = lambda x, y, z, t: k / (c * rol)

    # 物理场
    exist_heat_source = True
    heat_source = 1000  # w/m^3
    # ---> 表示单位体积内部热源的产热率,不知道这里怎么转化,从物理环境转化为偏微分方程
    f = lambda x, y, z, t: -200 * np.exp(x + y + z + t) / (c * rol)

    # 初始条件
    T0 = lambda x, y, z: np.exp(x + y + z)
    viewer.plot_cube_T(T0, show_grid=True)


    # 边界条件
    def G(x, y, z, t):
        '''
        Dirichlet boundary condition
        '''
        if x == 0:
            return np.exp(t + y + z)
        if x == 1:
            return np.exp(1 + y + z + t)
        if y == 0:
            return np.exp(t + x + z)
        if y == 1:
            return np.exp(1 + t + x + z)
        if z == 0:
            return np.exp(t + x + y)
        if z == 1:
            return np.exp(1 + t + x + y)
        return 'No limit'


    # 设置PDE
    question = Question_Dirichlet_3D_t()
    question.range = (x_range, y_range, z_range)
    question.C = a_square
    question.F = f
    question.T0 = T0
    question.G = G

    # 设置时间步
    t_range = (0, 2)
    question.t_range = t_range
    h_t = 0.1

    # 设置PDE求解器
    solver = Solver_t(question=question, h_Mesh=(h_mesh, h_mesh, h_mesh), h_Finite=(h_mesh, h_mesh, h_mesh), h_t=h_t,
                      scheme=1 / 2)
    solver.setBasisFunction(trialBasisFunctionType=BasisFunctionType.Three_Dimensional_Linear_Element,
                            testBasisFunctionType=BasisFunctionType.Three_Dimensional_Linear_Element)
    solver.setIntegrator(IntegratorType.Gauss)
    solver.integrator.set_gauss_point_number(8)
    solver.setPT_PbTb(PMatrixType.Three_Dimensional_Linear_Node, TMatrixType.Three_Dimensional_Linear_Cell,
                      PMatrixType.Three_Dimensional_Linear_Node, TMatrixType.Three_Dimensional_Linear_Cell)


    # 求解PDE
    process_drawing = True
    if process_drawing:
        solver.init_A_M()
        for step in range(0, solver.m):
            X = solver.step_t(step)
            # print(X)
            viewer.plot_cube_step(t=t_range[0] + h_t * step, T=X, show_grid=False)
        viewer.plot_cube_result(T=solver.X[-1])
    else:
        solver.makeLinearSystem()
        viewer.plot_cube_result(T=solver.X[-1])

    # 评估
    evaluate = Evaluate_t(solver)
    print('loo_norm_error = ', evaluate.Loo_norm_error())

    # result2VTK
    for time in range(solver.m):
        file = FileVTK()
        result=solver.X
        nodes = solver.PT_matrix.P
        cells = solver.PT_matrix.T
        for i in range(solver.PT_matrix.Nm):
            file.addPoint(nodes[:,i])

        for i in range(solver.PT_matrix.N):
            file.addCell(cells[:,i],12)

        for i in range(solver.PT_matrix.N):
            cell_data_i = 0.0
            for node_index in cells[:, i]:
                cell_data_i += result[time][node_index]
            cell_data_i/=len(cells[:, i])
            file.addCellData(cell_data_i)

        file.write2vtk('./result/result'+str(time)+'.vtk')