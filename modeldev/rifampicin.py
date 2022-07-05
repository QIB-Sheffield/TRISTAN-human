"""
j = jmax(1-exp(-a*c))
k = (jmax/c)(1-exp(-a*c))

small c:
j = jmax a c
k = jmax a

k(k0,a,c) = k0 * (1-exp(-a*c)) / (a*c) 

dj/dc = jmax a exp(-ac) = a (jmax - j)


Two compartments e

dnp/dt = J - A(1-exp(-a*np))
dne/dt = A(1-exp(-a*np)) - B(1-exp(-b*ne))

dnp/dt = J - k(kp,rp,np)*np - krenal*np
dne/dt = k(kp,rp,np)*np - k(kp,rp,np)*ne

np(t+dt) = np(t) + dt*J(t) - dt*k(kp,rp,np)*np
ne(t+dt) = ne(t) + dt*k(kp,rp,np)*np - dt*k(kp,rp,np)*ne
"""

import numpy as np
import matplotlib.pyplot as plt
import dcmri

def rate(k0, a, c):
    if isinstance(c, np.ndarray):
        k = np.empty(len(c))
        k.fill(k0)
        i = np.nonzero(c)
        k[i] = k0 * (1-np.exp(-a*c[i])) / (a*c[i]) 
        return k
    else:
        if c == 0:
            return k0
        else:
            return k0 * (1-np.exp(-a*c)) / (a*c)

def kinetics_2cfm(J, k, a, B, b):
    pass

def plot_k():

    cmax = 1
    dc = 0.01
    c = np.arange(0, cmax, dc)
    k = rate(3, 2, c)

    plt.title('Rate constant')
    plt.xlabel('concentration (mM)')
    plt.ylabel('rate constant')
    plt.plot(c, k, 'g-', label='k-model')
    plt.legend()
    plt.show()

# constants in natural units
tacq = 4 # hrs (total observation time)
dt = 1.0 # sec (internal resolution)

# rifampicin administration in blood
dose = 600 # mg
MTTinj = 90 # min
dispersion = 3

# rifampicin rate constants
kc = 0.0008 # /sec native elimination rate from blood to hepatocytes
ke = 0.00008 # /sec native elimination rate from hepatocytes to bile
kr = kc # /sec renal elimination rate
rc = 1.0 # /mg/l    initial decline in elimination rate
re = 1.0 # /mg/l    initial decline in elimination rate

# body parameters
vb = 6.0 # volume of blood in the body (litres)
ve = 0.6 # volume of hepatocytes in the body (liters)

# convert to standard units
tacq *= 60*60 # sec 
MTTinj *= 60 # sec


t = np.arange(0, tacq, dt)
J = dcmri.chain_propagator(t, MTTinj, dispersion) # /sec
J *= dose/np.trapz(J,t)     # mg/sec
# plt.plot(t/60/60, J, 'g-', label='influx')
# plt.show()

nt = len(t)
cc = np.empty(nt) # mg/l
ce = np.empty(nt) # mg/l

cc[0] = 0
ce[0] = 0
for k in range(nt-1):
    cc[k+1] = cc[k] + (dt/vb)*J[k] - dt*rate(kc,rc,cc[k])*cc[k] - dt*kr*cc[k]
    ce[k+1] = ce[k] + (dt/ve)*rate(kc,rc,cc[k])*vb*cc[k] - dt*rate(ke,re,ce[k])*ce[k]

plt.title('Concentrations')
plt.xlabel('time (hrs)')
plt.ylabel('concentration (mg/l)') # mg/l = ug/ml
plt.plot(t/60/60, cc, 'r-', label='blood')
plt.plot(t/60/60, ce, 'b-', label='hepatocytes')
plt.legend()
plt.show()

plt.title('Rate constants')
plt.xlabel('time (hrs)')
plt.ylabel('rate constant (/sec)')
plt.plot(t/60/60, rate(kc,rc,cc), 'r-', label='uptake')
plt.plot(t/60/60, rate(ke,re,ce), 'b-', label='excretion')
plt.legend()
plt.show()





