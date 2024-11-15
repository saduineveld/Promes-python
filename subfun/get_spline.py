# Get spline, taking policy, and grid as inputs

import numpy as np
from scipy.interpolate import RegularGridInterpolator

def get_grid_pol(y_col,xx_grid):
    nn = np.shape(xx_grid)[0]     
    if nn == 1:
        y_grid = y_col 
    elif nn > 1:
        y_grid = np.reshape(y_col,np.shape(xx_grid)[1:])   
    else:
        print("More than 4 dimensions not implemented")
    return y_grid

def constr_spline(y_grid,grid_vecs):
    nn = len(grid_vecs)
    if nn > 1 and nn < 5:
        pol = RegularGridInterpolator(grid_vecs,y_grid,"cubic",bounds_error=False,fill_value=None)
    else:
        print("More than 4 dimensions not implemented")
    return pol

def get_spline(y_col,xx_grid,grid_vecs):
    y_grid = get_grid_pol(y_col,xx_grid)
    pol = constr_spline(y_grid,grid_vecs)
    return pol
    
    
