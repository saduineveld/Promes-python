# RBC_noq model (no quadrature) module

import numpy as np

def get_res(alpha,beta,chi,delta,eta,nu,rho_z,xx,pol_old,lc_pol):
    ct = np.exp(lc_pol)
    kt = np.exp(xx[:,0])#First column
    zt = np.exp(xx[:,1])#Second column
    ht = labour(alpha,chi,eta,nu,zt,kt,ct)
    # Next period (n)
    kn = knext(alpha,delta,zt,kt,ht,ct)
    zn = np.exp(rho_z*xx[:,1])
    #Construct column vectors:
    lkn = np.log(kn[:,None])
    lzn = np.log(zn[:,None])
    cn = np.exp(pol_old(np.concatenate((lkn,lzn),axis=1)))
    hn = labour(alpha,chi,eta,nu,zn,kn,cn)
    rn = prod(alpha,zn,kn,hn)[1]    
    RES = beta*cn**-nu*(rn+1-delta)/(ct**-nu) - 1
    return RES


def get_omega(alpha,beta,delta,zss):
    omega = (1-beta*(1-delta))/(alpha*beta*zss)
    return omega

def get_kss_norm(alpha,beta,delta,zss):# Normalized hss=1
    omega = get_omega(alpha,beta,delta,zss)
    kss = omega**(1/(alpha-1))
    return kss

def get_chi(alpha,beta,delta,nu,zss):
    omega = get_omega(alpha,beta,delta,zss)
    kss = omega**(1/(alpha-1))
    css = get_css(zss,omega,delta,kss)
    chi = (1-alpha)*css**-nu*zss*kss**alpha
    return chi

def get_kss(alpha,beta,chi,delta,eta,nu,zss):
    omega = get_omega(alpha,beta,delta,zss)
    tmp = ((1-alpha)/chi*zss*(zss*omega-delta)**-nu)**eta
    kss = (tmp*omega**((alpha*eta+1)/(alpha-1)))**(1/(1+eta*nu))
    css = (zss*omega-delta)*kss
    hss = omega**(1/(1-alpha))*kss
    return kss,css,hss

def get_css(zss,omega,delta,kss):
    css = (zss*omega-delta)*kss
    return css


def marg_ut(nu,cc):
    dudc = cc**-nu
    return dudc

def labour(alpha,chi,eta,nu,zt,kt,ct):
    ht = ((1-alpha/chi)*ct**-nu*zt*kt**alpha)**(eta/(1+alpha*eta))
    return ht

def knext(alpha,delta,zt,kt,ht,ct):
    kn = prod(alpha,zt,kt,ht)[0] + (1-delta)*kt - ct
    return kn

def cons(alpha,delta,zt,kt,ht,kn):
    ct = prod(alpha,zt,kt,ht) + (1-delta)*kt - kn
    return ct

def prod(alpha,zt,kt,ht): 
    yt = zt*kt**alpha*ht**(1-alpha)
    rt = alpha*yt/kt
    wt = (1-alpha)*yt/ht
    return yt,rt,wt