#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
# import scikits.bvp1lg as bvp
import math,os
import time
import matplotlib
import potential as pot
from matplotlib import gridspec
import pylab
import matplotlib.cbook as cbook
os.environ['PATH'] = os.environ['PATH'] + '/usr/bin/latex'
# get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
plt.ion()

import scipy.interpolate
from matplotlib import colors

from scipy.integrate import quad
from scipy.misc import derivative

matplotlib.rcParams.update({'font.size': 14})
# matplotlib.rcParams["text.latex.preamble"].append(r'\usepackage[dvips]{graphicx}\usepackage{xfrac}')
plt.rc('text', usetex=True)
plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})


# In[3]:



r,phi,phi0,phis,dphi=pot.potential(R1=2,R2=4.0)


# In[ ]:





# In[214]:


def F1(r,rho1,rho2,R1,R2):
    return -rho2*(1+R2)*np.exp(-R2)*np.sinh(r)/r + (rho2-rho1)*(R1/r)*(np.sinh(r)/np.sinh(R1))*(1+np.exp(-R1)*((np.sinh(R1)/R1)-np.cosh(R1)))+rho1
    
def F2(r,rho1,rho2,R1,R2):
    return -rho2*(1+R2)*np.exp(-R2)*np.sinh(r)/r +   (rho2-rho1)*(np.sinh(R1)-R1*np.cosh(R1))*np.exp(-r)/r+rho2

def F3(r,rho1,rho2,R1,R2):
    return np.exp(-r)*((rho2-rho1)*(np.sinh(R1)-R1*np.cosh(R1))
                -rho2*(np.sinh(R2)-R2*np.cosh(R2)))/r
    
def funct(r,rho1,rho2,R1,R2):
    return np.piecewise(r,[r<R1,(r < R2) & (r>R1),r>R2],
                        [lambda r: F1(r,rho1,rho2,R1,R2),
                         lambda r: F2(r,rho1,rho2,R1,R2),
                         lambda r: F3(r,rho1,rho2,R1,R2)])


# In[ ]:





# In[ ]:





# In[ ]:





# In[10]:


def gamma(R,rho):
    return -R*(1.-np.cosh(-rho*(1+R)*np.exp(-R)+rho)+rho*(-rho*(1.+R)*np.exp(-R)+rho+rho/R*np.exp(-R)*(np.sinh(R)-R*np.cosh(R))))


# In[ ]:





# In[8]:


def gamma_num(r,phi0,phis,rho):
    return -r*(1.-np.cosh(phi0)+rho*(phi0-phis))


# In[213]:


def gamma_kR(charge=0.1):
#     rhos = np.array([0.1,0.2,0.3])
    kR   = np.linspace(0.2,10,50)
    r     = np.zeros((len(kR),32000))     # coordinate
    phi   = np.zeros((len(kR),32000))     # potential
    dphi   = np.zeros((len(kR),32000))     # electric field
    phis  = np.zeros(len(kR))             # surface potential
    phi0  = np.zeros(len(kR))             # mid-point potential
    # phi_DH= np.zeros((len(kR),32000))     # surface potential 
    # PN_DH= np.zeros((len(sigma),32000))     # surface potential 
    # PN= np.zeros((len(sigma),32000))     # normal pressure
    # hyd_p = np.zeros((len(sigma),32000))     # hydrostatic pressure 
    for i in range(len(kR)):
        pot.rho1 = charge
        pot.rho2 = charge
        pot.R1=kR[i]/2.; pot.R2=kR[i]
        r, phi[i] , phi0[i], phis[i], dphi[i]  = pot.potential(kR[i]/2.,kR[i],spherical=1)

    #         phi[i]=np.array(phi[i])-phis[i]*np.ones_like(np.array(phi[i]))
    return r, phi, phis, phi0 , kR

def gamma_rho(kR=3.0):
#     rhos = np.array([0.1,0.2,0.3])
    chg   = np.linspace(0.1,7,100)
    r     = np.zeros((len(chg),32000))     # coordinate
    phi   = np.zeros((len(chg),32000))     # potential
    dphi   = np.zeros((len(chg),32000))     # electric field
    phis  = np.zeros(len(chg))             # surface potential
    phi0  = np.zeros(len(chg))             # mid-point potential
    # phi_DH= np.zeros((len(kR),32000))     # surface potential 
    # PN_DH= np.zeros((len(sigma),32000))     # surface potential 
    # PN= np.zeros((len(sigma),32000))     # normal pressure
    # hyd_p = np.zeros((len(sigma),32000))     # hydrostatic pressure 
    for i in range(len(chg)):
        pot.rho1 = chg[i]
        pot.rho2 = chg[i]
        pot.R1=kR/2.; pot.R2=kR
        r, phi[i] , phi0[i], phis[i], dphi[i]  = pot.potential(kR/2.,kR,spherical=1.)

    #         phi[i]=np.array(phi[i])-phis[i]*np.ones_like(np.array(phi[i]))
    return r, phi, phis, phi0 , chg
    
   


# In[212]:


"""Dependence between surface tension and microgel's size """

plt.ylim(-1.0,0.0)
plt.xlim(0,10)
rho1=0.25
r, phi, phis, phi0 , kR = gamma_kR(rho1)
plt.plot(kR,gamma_num(kR,phi0,phis,rho1),'^r-',markevery=2,ms=5.)


rho1=0.75
r, phi, phis, phi0 , kR = gamma_kR(rho1)
plt.plot(kR,gamma_num(kR,phi0,phis,rho1),'go-',markevery=2,ms=5.)


rho1=1.5
r, phi, phis, phi0 , kR = gamma_kR(rho1)
plt.plot(kR,gamma_num(kR,phi0,phis,rho1),'bs-',markevery=2,ms=5.)

# rho1=1.5
# r, phi, phis, phi0 , kR = gamma_kR(rho1)
# plt.plot(kR,gamma_num(kR,phi0,phis,rho1),'^m')

plt.text(6.,-0.13,r'$\rho = 0.25$')
plt.text(6.,-0.35,r'$\rho = 0.75$')
plt.text(6.,-0.9,r'$\rho = 1.5$')
plt.ylabel(r'$\gamma$')
plt.xlabel(r'$\kappa R$')

plt.tight_layout()
plt.savefig('gamma_r.eps', format='eps',bbox_inches='tight')
plt.show()

# for ri in r:
# #     plt.plot(ri,gamma_num(ri,0.1),'go')
# #     plt.plot(ri,gamma_num(ri,0.2),'r^')
#     plt.plot(ri,gamma_num(ri,1.9),'bs-')


# In[ ]:





# In[211]:


"""Dependence between surface tension and charge density """
R=np.linspace(0.001, 15, 1000) 
plt.xlim(0,2.5)
plt.ylim(-0.4,0.)
kR=1.0
r, phi, phis, phi0 , chg = gamma_rho(kR)
plt.plot(chg,gamma_num(kR,phi0,phis,chg),'^r-',ms=5.)


kR=3.0
r, phi, phis, phi0 , chg = gamma_rho(kR)
plt.plot(chg,gamma_num(kR,phi0,phis,chg),'go-',ms=5.)


kR=10.0
r, phi, phis, phi0 , chg = gamma_rho(kR)
plt.plot(chg,gamma_num(kR,phi0,phis,chg),'bs-',ms=5.)


# plt.plot(chg,chg**2,'k--')

# plt.xscale('log')
# plt.yscale('log')
plt.text(0.85,-0.05,r'$\kappa R = 1$')
plt.text(0.75,-0.2,r'$\kappa R = 3$')
plt.text(0.4,-0.35,r'$\kappa R = 10$')
plt.ylabel(r'$\gamma$')
plt.xlabel(r'$\rho$')

plt.tight_layout()
plt.savefig('gamma_rho.eps', format='eps',bbox_inches='tight')
plt.show()


# In[209]:


def chg_d(r,rho1,rho2,R1,R2):
    return np.piecewise(r,[r<R1,(r < R2) & (r>R1),r>R2],
                        [lambda r: rho1,
                         lambda r: rho2,
                         lambda r: 0.])


# In[225]:


pot.rho1=1.0
pot.rho2=1.0
R=3.0
pot.R1=R; pot.R2=R

# Spherical
r, phi, phi0, phis , dphi = pot.potential(R,R,spherical=1.)
hyd_p_spher, P_N_spher, P_T_spher = pot.Pressure(r,phi,dphi)

print 2.*gamma_num(R,phi0,phis,1.0)/R
# Planar
r, phi, phi0, phis , dphi = pot.potential(R,R,spherical=0.)
hyd_p_plate, P_N_plate, P_T_plate = pot.Pressure(r,phi,dphi)
print 2.*gamma_num(R,phi0,phis,1.0)/R

print hyd_p_spher[0] - hyd_p_spher[-1]
print hyd_p_plate[0] - hyd_p_plate[-1]

# plot
plt.plot(r,hyd_p_spher,'^r-',markevery=1000, ms=5)
plt.plot(r,hyd_p_plate,'go-',markevery=1000, ms=5)
plt.xlim(0,10)
plt.text(5.5,0.28,r'spherical')
plt.text(5.5,0.02,r'planar')
plt.text(7.,0.15,r'$\kappa R = 3$')
plt.ylabel(r'$p$')
plt.xlabel(r'$\kappa R$')

plt.tight_layout()
plt.savefig('spherical_planar.eps', format='eps',bbox_inches='tight')
plt.show()


# In[156]:


plt.xlim(0,2)


pot.rho1=0.5
pot.rho2=0.5
R=3.0
pot.R1=R; pot.R2=R

r, phi, phi0, phis , dphi = pot.potential(R,R,spherical=1.)
hyd_p_spher, P_N_spher, P_T_spher = pot.Pressure(r,phi,dphi)
plt.plot(r/R,hyd_p_spher, '^r-',markevery=1000, ms=5)

pot.rho1=1.0
pot.rho2=1.0

r, phi, phi0, phis , dphi = pot.potential(R,R,spherical=1.)
hyd_p_spher, P_N_spher, P_T_spher = pot.Pressure(r,phi,dphi)
plt.plot(r/R,hyd_p_spher, 'go-',markevery=1000, ms=5)

pot.rho1=1.5
pot.rho2=1.5

r, phi, phi0, phis , dphi = pot.potential(R,R,spherical=1.)
hyd_p_spher, P_N_spher, P_T_spher = pot.Pressure(r,phi,dphi)
plt.plot(r/R,hyd_p_spher, 'bs-',markevery=1000, ms=5)

plt.text(1.5,0.57,r'$\rho = 1.5$')
plt.text(1.5,0.3,r'$\rho = 1.0$')
plt.text(1.5,0.1,r'$\rho = 0.5$')
plt.text(0.25,0.5,r'$\kappa R = 3$')
plt.ylabel(r'$p$')
plt.xlabel(r'$r/R$')

plt.tight_layout()
plt.savefig('hyd_pressure.eps', format='eps',bbox_inches='tight')
plt.show()


# In[220]:


plt.xlim(0,2)


pot.rho1=0.5
pot.rho2=0.5
R=3.0
pot.R1=R; pot.R2=R

r, phi, phi0, phis , dphi = pot.potential(R,R,spherical=1.)
hyd_p_spher, P_N_spher, P_T_spher = pot.Pressure(r,phi,dphi)
plt.plot(r/R,P_N_spher, '^r-',markevery=1000, ms=5)

pot.rho1=1.0
pot.rho2=1.0

r, phi, phi0, phis , dphi = pot.potential(R,R,spherical=1.)
hyd_p_spher, P_N_spher, P_T_spher = pot.Pressure(r,phi,dphi)
plt.plot(r/R,P_N_spher, 'go-',markevery=1000, ms=5)

pot.rho1=1.5
pot.rho2=1.5

r, phi, phi0, phis , dphi = pot.potential(R,R,spherical=1.)
hyd_p_spher, P_N_spher, P_T_spher = pot.Pressure(r,phi,dphi)
plt.plot(r/R,P_N_spher, 'bs-',markevery=1000, ms=5)

plt.text(1.5,0.55,r'$\rho = 1.5$')
plt.text(1.5,0.3,r'$\rho = 1.0$')
plt.text(1.5,0.1,r'$\rho = 0.5$')
plt.text(0.25,0.5,r'$\kappa R = 3$')
plt.ylabel(r'$P_N$')
plt.xlabel(r'$r/R$')

plt.tight_layout()
plt.savefig('P_N_dist.eps', format='eps',bbox_inches='tight')
plt.show()

plt.xlim(0,2)



# In[189]:


plt.xlim(0,2)


pot.rho1=0.5
pot.rho2=0.5
R=3.0
pot.R1=R; pot.R2=R

r, phi, phi0, phis , dphi = pot.potential(R,R,spherical=1.)
hyd_p_spher, P_N_spher, P_T_spher = pot.Pressure(r,phi,dphi)
plt.plot(r/R,P_T_spher, '^r-',markevery=1000, ms=5)

pot.rho1=1.0
pot.rho2=1.0

r, phi, phi0, phis , dphi = pot.potential(R,R,spherical=1.)
hyd_p_spher, P_N_spher, P_T_spher = pot.Pressure(r,phi,dphi)
plt.plot(r/R,P_T_spher, 'go-',markevery=1000, ms=5)

pot.rho1=1.5
pot.rho2=1.5

r, phi, phi0, phis , dphi = pot.potential(R,R,spherical=1.)
hyd_p_spher, P_N_spher, P_T_spher = pot.Pressure(r,phi,dphi)
plt.plot(r/R,P_T_spher, 'bs-',markevery=1000, ms=5)

plt.text(1.5,0.55,r'$\rho = 1.5$')
plt.text(1.5,0.3,r'$\rho = 1.0$')
plt.text(1.5,0.1,r'$\rho = 0.5$')
plt.text(0.25,0.5,r'$\kappa R = 3$')
plt.ylabel(r'$P_T$')
plt.xlabel(r'$r/R$')

plt.tight_layout()
plt.savefig('P_T_dist.eps', format='eps',bbox_inches='tight')
plt.show()



# In[227]:


plt.xlim(0,2)


pot.rho1=0.5
pot.rho2=0.5
R=5
pot.R1=R; pot.R2=R

r, phi, phi0, phis , dphi = pot.potential(R,R,spherical=1.)
hyd_p_spher, P_N_spher, P_T_spher = pot.Pressure(r,phi,dphi)
plt.plot(r/R,hyd_p_spher + 2.*gamma_num(R,phi0,phis,0.5)/R, '^r-',markevery=1000, ms=5)

R=20.0
pot.R1=R; pot.R2=R
r, phi, phi0, phis , dphi = pot.potential(R,R,spherical=1.)
hyd_p_spher, P_N_spher, P_T_spher = pot.Pressure(r,phi,dphi)
plt.plot(r/R,hyd_p_spher + 2.*gamma_num(R,phi0,phis,0.5)/R, 'go-',markevery=1000, ms=5)

R=10.
pot.R1=R; pot.R2=R
r, phi, phi0, phis , dphi = pot.potential(R,R,spherical=1.)
hyd_p_spher, P_N_spher, P_T_spher = pot.Pressure(r,phi,dphi)
plt.plot(r/R,hyd_p_spher + 2.*gamma_num(R,phi0,phis,0.5)/R, 'bs-',markevery=1000, ms=5)

# plt.text(1.5,0.57,r'$\rho = 1.5$')
# plt.text(1.5,0.3,r'$\rho = 1.0$')
# plt.text(1.5,0.1,r'$\rho = 0.5$')
plt.text(0.3,0.03,r'$\rho = 0.5$')
plt.ylabel(r'$p - p_\infty$')
plt.xlabel(r'$r/R$')

plt.tight_layout()
plt.savefig('dif_pressure_dist.eps', format='eps',bbox_inches='tight')
plt.show()


# In[2]:


import potential as pot
reload(pot)


# In[ ]:





# In[222]:


plt.xlim(0,10)


pot.rho1=0.5
pot.rho2=0.5
R=3.0
pot.R1=R; pot.R2=R

r, phi, phi0, phis , dphi = pot.potential(R,R,spherical=1.)
hyd_p_spher, P_N_spher, P_T_spher = pot.Pressure(r,phi,dphi)
plt.plot(r[1:],d_F(P_N_spher,r))
plt.plot(r,4*dphi*dphi/r)


# In[216]:


def d_F(F,r):
    return (F[1:]-F[0:-1])/(r[1:]-r[0:-1])


# In[196]:


(r[1:]-r[0:-1])


# In[217]:


d_F(P_N_spher,r)


# In[205]:





# In[ ]:




