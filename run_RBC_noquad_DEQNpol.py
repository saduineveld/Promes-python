# Run RBC model (without using quadrature)

import numpy as np
import math
from scipy.optimize import fsolve
from models import RBC_noquad_DEQNpol as RBC
from subfun import gridfun as gf
from subfun import get_spline as gs

import matplotlib.pyplot as plt

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
k_nodes = 7
lk_dev = 0.2
lk_inp =  [np.log(kss)-lk_dev,np.log(kss)+lk_dev,k_nodes]

lz_fac  = 2.6 # multiple of stnd. deviation
lz_std = math.sqrt(sigma_z**2 / (1-rho_z**2) )
z_nodes = 5
lz_inp = [-lz_fac*lz_std,lz_fac*lz_std,z_nodes]

# Get grid:
grid_input = np.array([lk_inp,lz_inp])
xx,xx_mat = gf.get_grid(grid_input)
grid_vecs = gf.get_vecs(grid_input)

# Initialize policy function
lkt = xx[:,0]#1 dim. vector (code line added for clarity)
lzt = xx[:,1]#1 dim. vector (code line added for clarity)
dcdk = 0.1#0.345
dcdz = 0.1#0.35
dhdk = -0.1#-0.54
dhdz = 0.1#0.48
lc_old = np.log(css)+dcdk*(lkt - np.log(kss)) + dcdz*lzt#col. vector
polc_old = gs.get_spline(lc_old,xx_mat,grid_vecs)
lh_old = np.log(hss)+dhdk*(lkt - np.log(kss)) + dhdz*lzt#col. vector (labor is decreasing in hours when nu > 1)
polh_old = gs.get_spline(lh_old,xx_mat,grid_vecs)

ly_old = np.concatenate((lc_old,lh_old),axis=0)
pol_old = (polc_old,polh_old)

# Create a new definition only for clarity
def equations(ly_pol,alpha,beta,chi,delta,eta,nu,rho_z,xx,pol_old):
    RES = RBC.get_res(alpha,beta,chi,delta,eta,nu,rho_z,xx,pol_old,ly_pol)
    return RES

#mm = len(lc_old)

# Set jacobian pattern (diagonal maxtrix of ones)
def jacobian_pattern(ly_old, *args):
    mm = np.round(len(ly_old)/2).astype(int)
    return np.tile(np.eye(mm),(2,2))

mm = len(lc_old)

# Solve current policy, given old policy (used in t+1), until convergence:
cnt = 0
while True:
    #ly_sol = fsolve(equations, ly_old, fprime=jacobian_pattern, args=(alpha,beta,chi,delta,eta,nu,rho_z,xx,pol_old), xtol=x_tol)
    ly_sol = fsolve(equations, ly_old, args=(alpha,beta,chi,delta,eta,nu,rho_z,xx,pol_old), xtol=x_tol)
    #print(lc_new)
    lc_sol = ly_sol[0:mm]  # update lc_old
    lh_sol = ly_sol[mm:]
    if not len(lc_old)== mm or not len(lh_old)== mm:
        print("length of policy is not correct")

    ly_old = ly_sol
    polc_old = gs.get_spline(lc_sol,xx_mat,grid_vecs)  # update pol_old
    polh_old = gs.get_spline(lh_sol,xx_mat,grid_vecs)  # update pol_old
    pol_old = (polc_old,polh_old)
    RES = equations(ly_old,alpha,beta,chi,delta,eta,nu,rho_z,xx,pol_old)
    cnt = cnt + 1
    if cnt == 1:
        print(RES)
    elif cnt % 1 == 0:
        print(RES) 

    if np.all(np.abs(RES) < max_error):
        break

# Rename solution:
pol = pol_old
pol_c = pol[0]
pol_h = pol[1]

## Plot policy function
# 1. construct grids


# A. lk = [lk1,...,lkn]; lz = zeros
# B. lk = lkss; lz=[lz1,...,lzn]

nodes_plt = 11
lk_A = np.reshape(np.linspace(np.log(kss)-lk_dev,np.log(kss)+lk_dev,nodes_plt),(-1,1))
lz_A = np.zeros(lk_A.shape)
xx_A = np.concatenate((lk_A,lz_A),axis=1)

lk_B = np.log(kss)*np.ones(lk_A.shape)
#lk_plot = np.concatenate((lk_A,lk_B),axis=0)
lz_B = np.reshape(np.linspace(-lz_fac*lz_std,lz_fac*lz_std,nodes_plt),(-1,1))
#lz_plot = np.concatenate((lz_A,lz_B),axis=0)
xx_B = np.concatenate((lk_B,lz_B),axis=1)

## 2. evaluate policy at grid
lc_A = pol_c(xx_A)
plt.plot(lk_A,lc_A)
plt.scatter(np.log(kss),np.log(css),c='red')
plt.show()

lc_B = pol_c(xx_B)
plt.plot(lz_B,lc_B)
plt.scatter(0,np.log(css),c='red')
plt.show()

lh_A = pol_h(xx_A)
plt.plot(lk_A,lh_A)
plt.scatter(np.log(kss),np.log(hss),c='red')
plt.show()

lh_B = pol_h(xx_B)
plt.plot(lz_B,lh_B)
plt.scatter(0,np.log(hss),c='red')
plt.show()

# Compute savings rate in steady state:
yss = RBC.prod(alpha,1,kss,hss)[0]
sss = 1-css/yss
print("Savings rate in steady state is ",sss)

# Compute implied savings rate policy:
#sav_rat(alpha,zt,kt,ht,ct): 
st_A = RBC.sav_rat(alpha,np.exp(lz_A),np.exp(lk_A),np.exp(lh_A),np.exp(lc_A))
plt.plot(lk_A,st_A)
plt.scatter(np.log(kss),sss,c='red')
plt.show()


