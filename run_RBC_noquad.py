# Run RBC model (without using quadrature)

import numpy as np
import math
from scipy.optimize import fsolve
from models import RBC_noquad as RBC
from subfun import gridfun_2D as gf
from subfun import get_spline_2D as gs

# Parameters
alpha = 0.36
beta = 0.985
delta = 0.025
nu = 2
eta = 4
rho_z = 0.95
sigma_z = 0.01
chi = 1
zss = 1

# Optimization parameters
max_error = 1e-8
x_tol     = 1e-10

# Get steady state:
kss,css,hss = RBC.get_kss(alpha,beta,chi,delta,eta,nu,zss)

# Construct grid vectors (keep lz=0 for now)
k_nodes = 5
lk_dev = 0.2
lk_inp =  [np.log(kss)-lk_dev,np.log(kss)+lk_dev,k_nodes]

lz_fac  = 2.6 # multiple of stnd. deviation
lz_std = math.sqrt(sigma_z**2 / (1-rho_z**2) )
z_nodes = 4
lz_inp = [-lz_fac*lz_std,lz_fac*lz_std,z_nodes]

# Get grid:
grid_input = np.array([lk_inp,lz_inp])
xx,xx_mat = gf.get_grid(grid_input)
grid_vecs = gf.get_vecs(grid_input)

# Initialize policy function
lkt = xx[:,0]#1 dim. vector (step added for clarity)
lzt = xx[:,1]#1 dim. vector (step added for clarity)
lc_old = np.log(css)+0.01*(lkt - np.log(kss)) + 0.01*lzt#col. vector
pol_old = gs.get_spline(lc_old,xx_mat,grid_vecs)

# Create a new definition only for clarity
def equations(lc_pol,alpha,beta,chi,delta,eta,nu,rho_z,xx,pol_old):
    RES = RBC.get_res(alpha,beta,chi,delta,eta,nu,rho_z,xx,pol_old,lc_pol)
    return RES

# Set jacobian pattern (diagonal maxtrix of ones)
def jacobian_pattern(lc_old, *args):
    return np.eye(len(lc_old))

# Solve current policy, given old policy (used in t+1), until convergence:
cnt = 0
while True:
    lc_sol = fsolve(equations, lc_old, fprime=jacobian_pattern, args=(alpha,beta,chi,delta,eta,nu,rho_z,xx,pol_old), xtol=x_tol)
    #print(lc_new)
    lc_old = lc_sol  # update lc_old
    pol_old = gs.get_spline(lc_sol,xx_mat,grid_vecs)  # update pol_old
    RES = equations(lc_old,alpha,beta,chi,delta,eta,nu,rho_z,xx,pol_old)
    cnt = cnt + 1
    print(RES)
    if np.all(np.abs(RES) < max_error):
        break

# Rename solution:
pol = pol_old
