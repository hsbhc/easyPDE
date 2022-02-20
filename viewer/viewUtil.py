import matplotlib
import numpy as np
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

class Viewer():
    def __init__(self,x=(0,1), y=(0,1), z=(0,1),color='lavender',edgecolor='dimgray'):
        self.x=x
        self.y=y
        self.z=z
        self.color=color
        self.edgecolor=edgecolor

    def _draw_surface(self,ax,xx,yy,zz):
        xx2, yy2 = np.meshgrid(xx, yy)
        ax.plot_surface(xx2, yy2, np.full_like(yy2, self.z[0]), color=self.color, edgecolor=self.edgecolor)
        ax.plot_surface(xx2, yy2, np.full_like(yy2, self.z[1]), color=self.color, edgecolor=self.edgecolor)

        yy2, zz2 = np.meshgrid(yy, zz)
        ax.plot_surface(np.full_like(yy2, self.x[0]), yy2, zz2, color=self.color, edgecolor=self.edgecolor)
        ax.plot_surface(np.full_like(yy2, self.x[1]), yy2, zz2, color=self.color, edgecolor=self.edgecolor)

        xx2, zz2 = np.meshgrid(xx, zz)
        ax.plot_surface(xx2, np.full_like(xx2, self.y[0]), zz2, color=self.color, edgecolor=self.edgecolor)
        ax.plot_surface(xx2, np.full_like(xx2, self.y[1]), zz2, color=self.color, edgecolor=self.edgecolor)

    def plot_cube(self):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        xx = np.linspace(self.x[0],self. x[1], 2)
        yy = np.linspace(self.y[0], self.y[1], 2)
        zz = np.linspace(self.z[0], self.z[1], 2)
        self._draw_surface(ax,xx,yy,zz)
        plt.title("geometry")
        plt.show()

    def plot_cube_mesh(self,h_mesh=0.5,show_node=True):
        self.h_mesh=0.5
        self.P = _make_Three_Dimensional_Linear_Node((self.x, self.y, self.z), (h_mesh, h_mesh, h_mesh))
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        xx = np.linspace(self.x[0],self. x[1], int((self.x[1] - self.x[0]) / h_mesh) + 1)
        yy = np.linspace(self.y[0], self.y[1], int((self.y[1] - self.y[0]) / h_mesh) + 1)
        zz = np.linspace(self.z[0], self.z[1], int((self.z[1] - self.z[0]) / h_mesh) + 1)
        self._draw_surface(ax,xx,yy,zz)
        if show_node:
            ax.scatter3D(self.P[0], self.P[1], self.P[2], c=self.edgecolor)
        plt.title("geometry -- mesh")
        plt.show()

    def plot_cube_T(self,T,show_grid=False):

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        if show_grid:
            xx = np.linspace(self.x[0],self. x[1], int((self.x[1] - self.x[0]) / self.h_mesh) + 1)
            yy = np.linspace(self.y[0], self.y[1], int((self.y[1] - self.y[0]) / self.h_mesh) + 1)
            zz = np.linspace(self.z[0], self.z[1], int((self.z[1] - self.z[0]) / self.h_mesh) + 1)
            self._draw_surface(ax,xx,yy,zz)
        nodes = ax.scatter3D(self.P[0], self.P[1], self.P[2], c=T(self.P[0],self.P[1],self.P[2]),cmap=cm.Reds,vmin=0,vmax=60)
        plt.colorbar(nodes)
        plt.title("geometry -- T0")
        plt.show()

    def plot_cube_step(self,t,T,show_grid=False):
        if t==0:
            self.fig = plt.figure()
        plt.clf()
        ax = self.fig.add_subplot(1, 1, 1, projection='3d')
        if show_grid:
            xx = np.linspace(self.x[0],self. x[1], int((self.x[1] - self.x[0]) / self.h_mesh) + 1)
            yy = np.linspace(self.y[0], self.y[1], int((self.y[1] - self.y[0]) / self.h_mesh) + 1)
            zz = np.linspace(self.z[0], self.z[1], int((self.z[1] - self.z[0]) / self.h_mesh) + 1)
            self._draw_surface(ax,xx,yy,zz)
        nodes = ax.scatter3D(self.P[0], self.P[1], self.P[2], c=T, cmap=cm.Reds,vmin=0,vmax=60)
        plt.colorbar(nodes)

        plt.title("geometry -- t = "+str(t))
        plt.draw()
        plt.pause(1)

    def plot_cube_result(self,T, show_grid=False):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        if show_grid:
            xx = np.linspace(self.x[0], self.x[1], int((self.x[1] - self.x[0]) / self.h_mesh) + 1)
            yy = np.linspace(self.y[0], self.y[1], int((self.y[1] - self.y[0]) / self.h_mesh) + 1)
            zz = np.linspace(self.z[0], self.z[1], int((self.z[1] - self.z[0]) / self.h_mesh) + 1)
            self._draw_surface(ax, xx, yy, zz)
        nodes = ax.scatter3D(self.P[0], self.P[1], self.P[2], c=T, cmap=cm.Reds,vmin=0,vmax=60)
        plt.colorbar(nodes)
        plt.title("geometry -- result ")
        plt.show()

def _make_Three_Dimensional_Linear_Node(ranges, h):
            '''
            Generate Three-dimensional linear nodes according to the range and step size
            Corresponding to P and Pb
            :param ranges: range such as ((0,1),(0,1),(0,1))
            :param h: Step size such as (0.5,0.5,0.5)
            :return: P: [[0.  0.  0.  0.5 0.5 0.5 1.  1.  1.  0.  0.  0.  0.5 0.5 0.5 1.  1.  1.  0.  0.  0.  0.5 0.5 0.5 1.  1.  1. ]
                        [0.  0.5 1.  0.  0.5 1.  0.  0.5 1.  0.  0.5 1.  0.  0.5 1.  0.  0.5 1.  0.  0.5 1.  0.  0.5 1.  0.  0.5 1. ]
                        [0.  0.  0.  0.  0.  0.  0.  0.  0.  0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 1.  1.  1.  1.  1.  1.  1.  1.  1. ]]
            '''
            N1 = int((ranges[0][1] - ranges[0][0]) / h[0])
            N2 = int((ranges[1][1] - ranges[1][0]) / h[1])
            N3 = int((ranges[2][1] - ranges[2][0]) / h[2])
            N1_m = N1 + 1
            N2_m = N2 + 1
            N3_m = N3 + 1
            Node_num = N1_m * N2_m * N3_m
            P = np.zeros((3, Node_num))
            for zn in range(N3_m):
                for rn in range(N1_m):
                    for cn in range(N2_m):
                        x = ranges[0][0] + rn * h[0]
                        y = ranges[1][0] + cn * h[1]
                        z = ranges[2][0] + zn * h[2]

                        j = zn * (N1_m * N2_m) + rn * N2_m + cn
                        P[0, j] = x
                        P[1, j] = y
                        P[2, j] = z
            return P

# viewer=Viewer()
# viewer.plot_cube()
# viewer.plot_cube_mesh(h_mesh=0.125)
# viewer.plot_cube_T(lambda x,y,z:np.exp(x+y+z))

# su=ax.plot_surface(X, Y, Z,facecolors=cm.Reds(Z)) facecolors=cm.Reds(np.full_like(zz2,np.exp(zz2+np.full_like(xx2, y[0]))))
#plot_cube(x=(0,1), y=(0,1), z=(0,1),h_mesh=0.125,show_node=True)
# plot_cube(x=(0,2), y=(0,2), z=(0,2),h_mesh=0.5)