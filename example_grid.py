import numpy as np
import math
from subfun import gridfun as gf


x1 =  [0,3,3]
x2 =  [10,30,5]
x3 =  [100,300,6]
x4 =  [1000,3000,4]

# 1D:
grid_input = np.array([x1])
#print("grid_input",grid_input)
xx,xx_grid = gf.get_grid(grid_input)
print("xx_grid",xx_grid)
print("xx",xx)#
print(xx_grid.shape)
print(xx_grid.shape[0])
#print(xx_grid.shape[1])
print(np.shape(xx_grid))  
print(np.shape(xx_grid)[0])   
print(xx_grid.ndim)
nn = xx_grid.ndim
print("nn",nn)
#print(np.shape(xx_grid)[1])  
#grid_vecs = gf.get_vecs(grid_input)

# 2D:
grid_input = np.array([x1,x2])
#print("grid_input",grid_input)
xx,xx_grid = gf.get_grid(grid_input)
print("xx_grid",xx_grid)
#print("xx",xx)#
print(xx_grid.shape)
nn = xx_grid.ndim
print("nn",nn)

# 3D:
grid_input = np.array([x1,x2,x3])
##print("grid_input",grid_input)
xx,xx_grid = gf.get_grid(grid_input)
#print("xx_grid",xx_grid)
print("xx",xx)#
print(xx.shape)
print(xx_grid.shape)
nn = xx_grid.ndim
print("nn",nn)

# 4D:
grid_input = np.array([x1,x2,x3,x4])
#print("grid_input",grid_input)
xx,xx_grid = gf.get_grid(grid_input)
#print("xx_grid",xx_grid)
#print("xx",xx)#
print(xx.shape)
#print(xx_grid.shape)
grid_vecs = gf.get_vecs(grid_input)

print(len(grid_vecs))
nn = xx_grid.ndim
print("nn",nn)
