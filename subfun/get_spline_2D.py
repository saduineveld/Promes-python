# Get spline, taking policy, and grid as inputs

import numpy as np
from scipy.interpolate import RegularGridInterpolator

def get_mat_pol(y_col,xx_mat):
    nn = np.shape(xx_mat)[0]
    print("nn",nn)
     
    if nn == 1:
        y_mat = y_col 
    elif nn == 2:
        nod1 = np.shape(xx_mat)[1]   
        nod2 = np.shape(xx_mat)[2]
        y_mat = np.reshape(y_col,(nod1,nod2))         
    elif nn > 2:
        print("More than 2 dimensions not implemented")

    return y_mat

def constr_spline(y_mat,grid_vecs):
    pol = RegularGridInterpolator((grid_vecs[0], grid_vecs[1]), y_mat,"cubic",bounds_error=False,fill_value=None)
    return pol

def get_spline(y_col,xx_mat,grid_vecs):
    y_mat = get_mat_pol(y_col,xx_mat)
    pol = constr_spline(y_mat,grid_vecs)
    return pol
    
    
