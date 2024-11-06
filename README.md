# Promes-python
Toolbox to solve infinite horizon DSGE models with time iteration and spline approximation

Currently (6.11.24) the toolbox is only suited for a 2-dimensional state space, and a 1-dimensional policy. It only includes two modules:
- gridfun_2D.py
- get_spline_2D.py

One simple example is included, which solves a standard RBC model without using any type of quadrature (it uses a simply point-estimate for future exogenous state variable).
