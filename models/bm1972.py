# RBC_noq model (no quadrature) module

import numpy as np

def get_res(alpha,beta,xx,pol_old,lc_pol):
    ct = np.exp(lc_pol)
    kt = np.exp(xx[:,0])#First column
    # Next period (n)
    kn = knext(alpha,kt,ct)
    #Construct column vectors:
    lkn = np.log(kn[:,None])
    cn = np.exp(pol_old(lkn))
    rn = prod(alpha,kn)[1]    
    RES = beta*cn**-1*rn/(ct**-1) - 1
    return RES


def get_kss(alpha,beta):
    kss = 1/(alpha*beta)**(1/(alpha-1))
    css = prod(alpha,kss) - kss
    return kss,css

def marg_ut(cc):
    dudc = cc**-1
    return dudc

def knext(alpha,kt,ct):
    kn = prod(alpha,kt)[0] - ct
    return kn

def cons(alpha,kt,kn):
    ct = prod(alpha,kt) - kn
    return ct

def prod(alpha,kt): 
    yt = kt**alpha
    rt = alpha*yt/kt
    return yt,rt