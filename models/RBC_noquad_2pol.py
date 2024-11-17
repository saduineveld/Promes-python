# RBC no quadrature, 2 policy model (no quadrature) module

import numpy as np

def get_res(alpha,beta,chi,delta,eta,nu,rho_z,xx,pol_old,ly_pol):
    mm = np.round(len(ly_pol)/2).astype(int)
    lc_pol = ly_pol[0:mm]
    lh_pol = ly_pol[mm:]

    if not len(lc_pol)== mm or not len(lh_pol)== mm:
        print("length of policy is not correct")

    ct = np.exp(lc_pol)
    ht = np.exp(lh_pol)
    lkt = xx[:,0]
    kt = np.exp(lkt)#First column
    lzt = xx[:,1]
    zt = np.exp(lzt)#Second column
    # LH = par.eta/(1+par.alpha*par.eta) * ( -log(par.chi) -par.nu*LC + log(1-par.alpha) + LZ + par.alpha*LK );
    RES2 = eta/(1+alpha*eta)*(-np.log(chi)-nu*lc_pol + np.log(1-alpha) + lzt + alpha*lkt) - lh_pol

    # Next period (n)
    kn = knext(alpha,delta,zt,kt,ht,ct)
    zn = np.exp(rho_z*xx[:,1])
    #Construct column vectors:
    lkn = np.log(kn[:,None])
    lzn = np.log(zn[:,None])
    polc_old = pol_old[0]
    polh_old = pol_old[1]
    cn = np.exp(polc_old(np.concatenate((lkn,lzn),axis=1)))
    #hn = np.exp(polh_old(np.concatenate((lkn,lzn),axis=1)))
    hn = labour(alpha,chi,eta,nu,zn,kn,cn)
    rn = prod(alpha,zn,kn,hn)[1]    
    RES1 = beta*cn**-nu*(rn+1-delta)/(ct**-nu) - 1    
    RES = np.concatenate((RES1,RES2),axis=0)
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
    ht = (((1-alpha)/chi)*ct**-nu*zt*kt**alpha)**(eta/(1+alpha*eta))
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