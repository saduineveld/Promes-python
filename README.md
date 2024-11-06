# Promes-python
Toolbox to solve infinite horizon DSGE models with time iteration and spline approximation
*Required pacakges for Python 3: numpy, scipy.interpolate, math*

Currently (6.11.24) the toolbox is only suited for a 2-dimensional state space, and a 1-dimensional policy. It only includes two modules:
- gridfun_2D.py
- get_spline_2D.py

One simple example is included, which solves a standard RBC model without using any type of quadrature (it uses a simply point-estimate for future exogenous state variable).

## Short Manual
* We added code from the example "run_RBC_no_quad.py" 
### Glossary
- $n$: number of dimensions of state space (maximum is currentl 2)
- $m$: total number of gridpoints
- grid: rectangular grid of state space

### Model file
* Example: `models/RBC_noquad_2D.py`*
- Inputs: old policy (`pol_old`, which is a spline constructed with `np.RegularGriddInterpolator`), policy at grid points (`lc_pol`), `xx` which contains all gridpoints ($m$m x $n$ array)
- Output: Euler residual vector (`RES`, length $m$)

### Set parameters and compute steady state
```
kss,css,hss = RBC.get_kss(alpha,beta,chi,delta,eta,nu,zss)
```

### Construct grid
1. Construct a 1-dimensional input array for each state variable with format:
```
xi_inp = [lower bound, upper bound, number of nodes]
```

2. Put these input vectors in one array:
```
grid_input = np.array([x1_inp,...,xn_inp])
```

3. Get `xx` ($m$ x $n$ array) containing all gridpoints, and get `xx_mat` ($n$ x $nod1$ x $nod2$ array), which contains the same gridpoints, but in a $n+1$ dimensional grid:
```
xx,xx_mat = gf.get_grid(grid_input)
```
4. Get grid vectors (tuple of $n$ vectors, created using linspace, with lower bound, upper bound and number of nodes):
```
grid_vecs = gf.get_vecs(grid_input)
```
### Set initial guess
*Recommendation: a linear approximation with a small coefficient but the correct sign usually works fine.*
1. Initialize the policy function as a 1 dimensional vector:
```
lc_old = np.log(css)+0.01*(xx[:,0] - np.log(kss)) + 0.01*xx[:,1]
```
2. Make spline `pol` from this guess:
```
pol_old = gs.get_spline(lc_old,xx_mat,grid_vecs)
```

### Solve policy function using time iteration
- *Optional: create easy to read definition for system of equations* 
- Set pattern of the Jacobian matrix (which tells `fsolve` that each equation is independent)
- Solve for the new policy at the gridpoints, given the old policy (used in t+1), until convergence:
```
while True:
    lc_new = fsolve(equations, lc_old, fprime=jacobian_pattern, args=(alpha,beta,chi,delta,eta,nu,rho_z,xx,pol_old), xtol=x_tol)
    #print(lc_new)
    lc_old = lc_new  # update lc_old
    pol_old = gs.get_spline(lc_old,xx_mat,grid_vecs)  # update pol_old
    RES = equations(lc_old,alpha,beta,chi,delta,eta,nu,rho_z,xx,pol_old)
    cnt = cnt + 1
    print(RES)
    if np.all(np.abs(RES) < max_error):
        break
```





