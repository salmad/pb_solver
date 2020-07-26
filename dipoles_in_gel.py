#!/usr/bin/env python
# coding: utf-8

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

matplotlib.rcParams.update({'font.size': 16})
# matplotlib.rcParams["text.latex.preamble"].append(r'\usepackage[dvips]{graphicx}\usepackage{xfrac}')
plt.rc('text', usetex=True)
plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})

import potential as pot
reload(pot)


# In[245]:


# Constant dipole, variable dipole concentration
p1      = 2.
cp1     = 2.
pot.rho1= 2.
pot.rho2= 2.
pot.R1  = 1.
pot.R2  = 10.
pot.c_p = cp1
pot.p   = p1
r1,phi1,phi01,phis1,dphi1=pot.potential(R1=1.0,R2=pot.R2)


p2      = 2.
cp2     = 1.
pot.rho1= 2.
pot.rho2= 2.0
pot.R1  = 1.
pot.R2  = 10.
pot.c_p = cp2
pot.p   = p2
r2,phi2,phi02,phis2,dphi2=pot.potential(R1=1,R2=pot.R2)


p3      = 2.
cp3     = 0.2
pot.rho1= 2.
pot.rho2= 2.
pot.R1  = 1.
pot.R2  = 10.
pot.c_p = cp3
pot.p   = p3
r3,phi3,phi03,phis3,dphi3=pot.potential(R1=1,R2=pot.R2)


# In[246]:


matplotlib.rcParams.update({'font.size': 18})
# matplotlib.rcParams["text.latex.preamble"].append(r'\usepackage[dvips]{graphicx}\usepackage{xfrac}')
plt.rc('text', usetex=True)
plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})

#1/ potential
plt.xlabel('$r$')
plt.ylabel('$\phi$')
plt.plot(r1-pot.R2,phi1,'r^-',markevery=1000,markeredgecolor='k')
# ,label="$c_{p,\infty} = 2.0$")
# plt.plot(-r1,phi1,'ro-',markevery=1000)
# plt.plot(r1,cp1*np.sinh(dphi1*p1)/(dphi1*p1),'r^-',markevery=1000)

plt.plot(r2-pot.R2, phi2, 'go-', markevery=1000,markeredgecolor='k')
# ,label="$c_{p,\infty} = 1.0$")
# plt.plot(-r2,phi2,'go-',markevery=1000)
# plt.plot(r2,cp2*np.sinh(dphi2*p2)/(dphi2*p2),'g^-',markevery=1000)

plt.plot(r3-pot.R2, phi3, 'bs-',markevery=1000,markeredgecolor='k')
# ,label="$c_{p,\infty} = 0.2$")
# plt.plot(-r3,phi3,'bo-',markevery=1000)

# plt.plot(r3,cp3*np.sinh(dphi3*p3)/(dphi3*p3),'b^-',markevery=1000)

plt.legend()
plt.xlim(-5.,5)
plt.tight_layout()
plt.savefig('potential_variable_cp.eps', format='eps',bbox_inches='tight')
plt.show()

#2/ Dipole concentration

plt.xlabel('$r$')
plt.ylabel('${c_d}/{c_{d,b}}-1$')
# plt.plot(r1,phi1,'ro-',markevery=1000)
plt.plot(r1-pot.R2,1*np.sinh(dphi1*p1)/(dphi1*p1)-1,'r^-',markevery=1000,markeredgecolor='k')
# plt.plot(-r1,1*np.sinh(dphi1*p1)/(dphi1*p1),'r^-',markevery=1000)

# plt.plot(r2,phi2,'go-',markevery=1000)
plt.plot(r2-pot.R2,1*np.sinh(dphi2*p2)/(dphi2*p2)-1,'g^-',markevery=1000,markeredgecolor='k')
# plt.plot(-r2,1*np.sinh(dphi2*p2)/(dphi2*p2),'g^-',markevery=1000)

# plt.plot(r3,phi3,'bo-',markevery=1000)
plt.plot(r3-pot.R2,1*np.sinh(dphi3*p3)/(dphi3*p3)-1,'b^-',markevery=1000,markeredgecolor='k')
# plt.plot(-r3,1*np.sinh(dphi3*p3)/(dphi3*p3),'b^-',markevery=1000)

plt.xlim(-5.,5)
plt.ylim(-.1,.5)

plt.tight_layout()
plt.savefig('dipole_conc_variable_cp.eps', format='eps',bbox_inches='tight')
plt.show()

#3/ electrolyte concentration

plt.xlabel('$r$')
plt.ylabel('$c_\pm/c_b$')

plt.plot(r1-pot.R2,np.exp(phi1),'r^-',markevery=1000,markeredgecolor='k')
plt.plot(r1-pot.R2,np.exp(-phi1),'r^-',markevery=1000,mfc='none')
# plt.plot(-r1,np.exp(phi1),'r^-',markevery=1000)
# plt.plot(r1, -np.sinh(phi1)+5.5* pot.delta(r1,1,0)+5.5* pot.delta(r1,4,1),'r^',markevery=1000)
# plt.plot(-r1,-np.sinh(phi1)+5.5* pot.delta(r1,1,0)+5.5* pot.delta(r1,4,1),'r^',markevery=1000)

plt.plot(r2-pot.R2,np.exp(phi2),'go-',markevery=1000,markeredgecolor='k')
plt.plot(r2-pot.R2,np.exp(-phi2),'go-',markevery=1000,mfc='none')
# plt.plot(-r2,np.exp(phi2),'go-',markevery=1000)
# plt.plot(r2,  -np.sinh(phi2)+5.5* pot.delta(r1,1,0)+5.5* pot.delta(r1,4,1),'g^',markevery=1000)
# plt.plot(-r2, -np.sinh(phi2)+5.5* pot.delta(r1,1,0)+5.5* pot.delta(r1,4,1),'g^',markevery=1000)

plt.plot(r3-pot.R2,np.exp(phi3),'bs-',markevery=1000,markeredgecolor='k')
plt.plot(r3-pot.R2,np.exp(-phi3),'bs-',markevery=1000,mfc='none')
# plt.plot(-r3,np.exp(phi3),'bo-',markevery=1000)
# plt.plot(r3,  -np.exp(phi3)+5.5* pot.delta(r1,1,0)+5.5* pot.delta(r1,4,1),'b^',markevery=1000)
# plt.plot(-r3, -np.exp(phi3)+5.5* pot.delta(r1,1,0)+5.5* pot.delta(r1,4,1),'b^',markevery=1000)

plt.xlim(-5.,5)
plt.ylim(0,5)

plt.tight_layout()
plt.savefig('electrolyte_conc_variable_cp.eps', format='eps',bbox_inches='tight')
plt.show()


# In[ ]:


#Constant dipole, variable dipole concentration
p1      = 2.
cp1     = 2.
pot.rho1= 2.
pot.rho2= 2.
pot.R1  = 1.
pot.R2  = 10.
pot.c_p = cp1
pot.p   = p1
r1,phi1,phi01,phis1,dphi1=pot.potential(R1=1.0,R2=pot.R2)


p2      = 2.
cp2     = 1.
pot.rho1= 2.
pot.rho2= 2.0
pot.R1  = 1.
pot.R2  = 10.
pot.c_p = cp2
pot.p   = p2
r2,phi2,phi02,phis2,dphi2=pot.potential(R1=1,R2=pot.R2)


p3      = 2.
cp3     = 0.2
pot.rho1= 2.
pot.rho2= 2.
pot.R1  = 1.
pot.R2  = 10.
pot.c_p = cp3
pot.p   = p3
r3,phi3,phi03,phis3,dphi3=pot.potential(R1=1,R2=pot.R2)


# In[205]:


def system_rho(kR=9.,dm = 1., cd = 1.):
#     rhos = np.array([0.1,0.2,0.3])
    chg   = np.linspace(0.1,7,200)
    r     = np.zeros((len(chg),32000))     # coordinate
    phi   = np.zeros((len(chg),32000))     # potential
    dphi   = np.zeros((len(chg),32000))     # electric field
    phis  = np.zeros(len(chg))             # surface potential
    phi0  = np.zeros(len(chg))             # mid-point potential
    # phi_DH= np.zeros((len(kR),32000))     # surface potential 
    # PN_DH= np.zeros((len(sigma),32000))     # surface potential 
    # PN= np.zeros((len(sigma),32000))     # normal pressure
    # hyd_p = np.zeros((len(sigma),32000))     # hydrostatic pressure 
    pot.c_p = cd
    pot.p   = dm
    for i in range(len(chg)):
        pot.rho1 = chg[i]
        pot.rho2 = chg[i]
        pot.R1=kR/2.; pot.R2=kR
        r, phi[i] , phi0[i], phis[i], dphi[i]  = pot.potential(R1=kR/2.,R2=kR)

        #         phi[i]=np.array(phi[i])-phis[i]*np.ones_like(np.array(phi[i]))
    return r, phi, phis, phi0 , chg, dphi


# In[206]:


r, phi, phis, phi0 , chg, dphi = system_rho()


# In[235]:


matplotlib.rcParams.update({'font.size': 18})
# matplotlib.rcParams["text.latex.preamble"].append(r'\usepackage[dvips]{graphicx}\usepackage{xfrac}')
plt.rc('text', usetex=True)
plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})

plt.plot(chg,phis,'bs-',markevery=10,markeredgecolor='k')
plt.plot(chg,phi0,'r^-',markevery=10,markeredgecolor='k')
plt.plot(0.2*chg,0.2*chg,'k--',markevery=1)
plt.plot(0.2*chg,0.2*chg/2.,'k--',markevery=1)

plt.plot(chg+1.65,np.log(2*(chg+1.65)),'k--',markevery=1)


plt.xlabel(r'$\rho$')
plt.ylabel(r'$\phi_s, \phi_0$')
plt.xlim(0,4)
plt.ylim(0,2.5)

plt.tight_layout()
plt.savefig('phis_phi0.pdf', format='pdf',bbox_inches='tight')
plt.show()


# In[218]:


def Gamma(dphi, r, p):
    cd = np.sinh(dphi[1:]*p)/(dphi[1:]*p)-1
    dr = r[1:]-r[:-1]
    return cd.sum()*dr[0]

def remove_nan(array):
    for i in range(len(array)):
        if np.isnan(array[i]) and i>0 and i<200:
            array[i] = ( array[i-1])
    return array


# In[219]:


def system_cd(kR=9.,dm = 1., chg = 1.):
#     rhos = np.array([0.1,0.2,0.3])
    cd   = np.linspace(0.1,7,200)
    r    = np.zeros((len(cd),32000))     # coordinate
    phi  = np.zeros((len(cd),32000))     # potential
    dphi = np.zeros((len(cd),32000))     # electric field
    phis = np.zeros(len(cd))             # surface potential
    phi0 = np.zeros(len(cd))             # mid-point potential
    # phi_DH= np.zeros((len(kR),32000))     # surface potential 
    # PN_DH= np.zeros((len(sigma),32000))     # surface potential 
    # PN= np.zeros((len(sigma),32000))     # normal pressure
    # hyd_p = np.zeros((len(sigma),32000))     # hydrostatic pressure 
    pot.p   = dm
    for i in range(len(cd)):
        pot.rho1 = chg
        pot.rho2 = chg
        pot.c_p = cd[i]
        pot.R1=kR/2.; pot.R2=kR
        r, phi[i] , phi0[i], phis[i], dphi[i]  = pot.potential(R1=kR/2.,R2=kR)

        #         phi[i]=np.array(phi[i])-phis[i]*np.ones_like(np.array(phi[i]))
    return r, phi, phis, phi0 , chg, dphi


# In[236]:


dm1=2.0; dm2=1.0; dm3=0.5;
cd1 = 1.0; cd2 =1.0; cd3 =1.0;
r1, phi1, phis1, phi01 , chg1, dphi1 = system_rho(dm=dm1,cd=cd1)
r2, phi2, phis2, phi02 , chg2, dphi2 = system_rho(dm=dm2,cd=cd2)
r3, phi3, phis3, phi03 , chg3, dphi3 = system_rho(dm=dm3,cd=cd3)


# In[237]:


chg   = np.linspace(0.1,7,200)

gamma1 = np.zeros_like(chg)
gamma2 = np.zeros_like(chg)
gamma3 = np.zeros_like(chg)

for i in range(len(dphi1)):
#     print Gamma(dphi[i], r, p1)
    gamma1[i] = cd1*Gamma(dphi1[i], r, dm1)
    gamma2[i] = cd2*Gamma(dphi2[i], r, dm2)
    gamma3[i] = cd3*Gamma(dphi3[i], r, dm3)
#     plt.scatter(chg[i],cd1*Gamma(dphi1[i], r, dm1),marker='^',c='r',edgecolors='k')

gamma1 = remove_nan(gamma1)
gamma2 = remove_nan(gamma2)
gamma3 = remove_nan(gamma3)
plt.plot(chg, gamma1, 'r^-', markevery=7,markeredgecolor='k')
plt.plot(chg, gamma2, 'bs-', markevery=7,markeredgecolor='k')
plt.plot(chg, gamma3, 'go-', markevery=7,markeredgecolor='k')
    
# for i in range(len(dphi2)):
# #     print Gamma(dphi[i], r, p1)
#     plt.scatter(chg[i],cd2*Gamma(dphi2[i], r, dm2),marker='s',c='b',edgecolors='k')
    
# for i in range(len(dphi3)):
# #     print Gamma(dphi[i], r, p1)
#     plt.scatter(chg[i],cd3*Gamma(dphi3[i], r, dm3),marker='o',c='g',edgecolors='k')

plt.ylim(-0.1,1.5)
plt.xlabel(r'$\rho$')
plt.ylabel(r'$\Gamma$')

# plt.xscale('log')
plt.tight_layout()
plt.savefig('gamma_vs_rho_diff_p.pdf', format='pdf',bbox_inches='tight')
plt.show()


# In[197]:





# In[238]:


dm1=2.0; dm2=1.0; dm3=0.5;
r1, phi1, phis1, phi01 , chg1, dphi1 = system_cd(dm=dm1,chg=2.0)
r2, phi2, phis2, phi02 , chg2, dphi2 = system_cd(dm=dm2,chg=2.0)
r3, phi3, phis3, phi03 , chg3, dphi3 = system_cd(dm=dm3,chg=2.0)


# In[239]:


#different dipole moments
cd   = np.linspace(0.1,7,200)

gamma1 = np.zeros_like(cd)
gamma2 = np.zeros_like(cd)
gamma3 = np.zeros_like(cd)

for i in range(len(dphi1)):
#     print Gamma(dphi[i], r, p1)
    gamma1[i] = cd[i]*Gamma(dphi1[i], r, dm1)
    gamma2[i] = cd[i]*Gamma(dphi2[i], r, dm2)
    gamma3[i] = cd[i]*Gamma(dphi3[i], r, dm3)
#     plt.scatter(chg[i],cd1*Gamma(dphi1[i], r, dm1),marker='^',c='r',edgecolors='k')

gamma1 = remove_nan(gamma1)
gamma2 = remove_nan(gamma2)
gamma3 = remove_nan(gamma3)
plt.plot(cd, gamma1, 'r^-', markevery=7,markeredgecolor='k')
plt.plot(cd, gamma2, 'bs-', markevery=7,markeredgecolor='k')
plt.plot(cd, gamma3, 'go-', markevery=7,markeredgecolor='k')

    
plt.xlabel(r'$c_{d,b}$')
plt.ylabel(r'$\Gamma$')
# plt.xscale('log')

plt.tight_layout()
plt.savefig('gamma_vs_cd_diff_p.pdf', format='pdf',bbox_inches='tight')
plt.show()


# In[240]:


###different charge
dm1=1.0; dm2=1.0; dm3=1.0;
chg1=4.0;chg2=2.0;chg3=0.5;
r1, phi1, phis1, phi01 , chg1, dphi1 = system_cd(dm=dm1,chg=chg1)
r2, phi2, phis2, phi02 , chg2, dphi2 = system_cd(dm=dm2,chg=chg2)
r3, phi3, phis3, phi03 , chg3, dphi3 = system_cd(dm=dm3,chg=chg3)


# In[244]:


#different mu-g charge

cd   = np.linspace(0.1,7,200)

gamma1 = np.zeros_like(cd)
gamma2 = np.zeros_like(cd)
gamma3 = np.zeros_like(cd)

for i in range(len(dphi1)):
#     print Gamma(dphi[i], r, p1)
    gamma1[i] = cd[i]*Gamma(dphi1[i], r, dm1)
    gamma2[i] = cd[i]*Gamma(dphi2[i], r, dm2)
    gamma3[i] = cd[i]*Gamma(dphi3[i], r, dm3)
#     plt.scatter(chg[i],cd1*Gamma(dphi1[i], r, dm1),marker='^',c='r',edgecolors='k')

gamma1 = remove_nan(gamma1)
gamma2 = remove_nan(gamma2)
gamma3 = remove_nan(gamma3)
plt.plot(cd, gamma1, 'r^-', markevery=7,markeredgecolor='k')
plt.plot(cd, gamma2, 'bs-', markevery=7,markeredgecolor='k')
plt.plot(cd, gamma3, 'go-', markevery=7,markeredgecolor='k')


plt.ylim(-.05,1)
plt.xlabel(r'$c_{d,b}$')
plt.ylabel(r'$\Gamma$')

plt.tight_layout()
plt.savefig('gamma_vs_cd_diff_rho.pdf', format='pdf',bbox_inches='tight')
plt.show()


# In[ ]:




