from Data2VTK import FileVTK

file=FileVTK()

file.addPoint([1,2,3])
file.addCell([1,2,3,4,5,6,7,8],8)
file.addCell([1,2,3,4,5,6,7,8],8)
file.addCellData(56.3)
file.addCellData(0.3223324)
file.write2vtk('test.vtk')