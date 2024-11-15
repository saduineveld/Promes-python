# Functions to construct grid

import numpy as np

#grid_input: nx3 matrix;
# each row: [lower bound, upper bound, number of nodes]
def get_vecs(grid_input):
    nn = np.shape(grid_input)[0]
    grid_vecs = []
    for ii in range (0,nn):
        nod = int(grid_input[ii,2])
        vec_temp = np.linspace(grid_input[ii,0],grid_input[ii,1],nod)
        grid_vecs.append(vec_temp)
    return grid_vecs

def get_grid(grid_input):
    nn = np.shape(grid_input)[0]
    if nn == 1:
        xx_grid = np.mgrid[grid_input[0,0]:grid_input[0,1]:(grid_input[0,2]*1j)]
    elif nn == 2:
        xx_grid = np.mgrid[grid_input[0,0]:grid_input[0,1]:(grid_input[0,2]*1j),
                    grid_input[1,0]:grid_input[1,1]:(grid_input[1,2]*1j)]         
    elif nn == 3:
        xx_grid = np.mgrid[grid_input[0,0]:grid_input[0,1]:(grid_input[0,2]*1j),
                    grid_input[1,0]:grid_input[1,1]:(grid_input[1,2]*1j),
                    grid_input[2,0]:grid_input[2,1]:(grid_input[2,2]*1j)]
    elif nn == 4:
        xx_grid = np.mgrid[grid_input[0,0]:grid_input[0,1]:(grid_input[0,2]*1j),
                    grid_input[1,0]:grid_input[1,1]:(grid_input[1,2]*1j),
                    grid_input[2,0]:grid_input[2,1]:(grid_input[2,2]*1j),
                    grid_input[3,0]:grid_input[3,1]:(grid_input[3,2]*1j)]
    else:
        print("More than 4 dimensions not implemented")
        
    #mm = np.prod(grid_input[:,[2]])
    if nn == 1:
        xx = np.reshape(xx_grid,(-1,1))
    else:
        for ii in range (0,nn):
            xi = xx_grid[ii]
            x_rsh = np.reshape(xi,(-1,1))
            if ii == 0:
                xx = x_rsh    
            else:
                xx = np.concatenate((xx,x_rsh),axis=1)

    return xx,xx_grid#xx are concatenated column vectors, xx_grid is meshgrid