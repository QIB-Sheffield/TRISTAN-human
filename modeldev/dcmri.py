import math
import numpy as np
from scipy.special import gamma
from scipy.interpolate import CubicSpline
from scipy.stats import rv_histogram
import matplotlib.pyplot as plt


def expconv(T, time, a):
    """Convolve a 1D-array with a normalised exponential.

    expconv() uses an efficient and accurate numerical formula to calculate the convolution,
    as detailed in the appendix of Flouri et al., Magn Reson Med, 76 (2016), pp. 998-1006.

    Note (1): by definition, expconv preserves the area under a(time)
    Note (2): if T=0, expconv returns a copy of a

    Arguments
    ---------
    a : numpy array
        the 1D array to be convolved.
    time : numpy array
        the time points where the values of ca are defined
        these do not have to to be equally spaced.
    T : float
        the characteristic time of the the exponential function.
        time and T must be in the same units.

    Returns
    -------
    a numpy array of the same shape as ca.

    Example
    -------
    coming soon..

    """
    if T==0: return a

    n = len(time)
    f = np.zeros(n)
    x = (time[1:n] - time[0:n-1])/T
    da = (a[1:n] - a[0:n-1])/x
    E = np.exp(-x)
    E0 = 1-E
    E1 = x-E0
    add = a[0:n-1]*E0 + da*E1
    for i in range(0,n-1):
        f[i+1] = E[i]*f[i] + add[i]      
    return f




def trapz(t, f):
    n = len(f)
    g = np.empty(n)
    g[0] = 0
    for i in range(n-1):
        g[i+1] = g[i] + (t[i+1]-t[i]) * (f[i+1]+f[i]) / 2
    return g

def utrapz(dt, f):
    n = len(f)
    g = np.empty(n)
    g[0] = 0
    for i in range(n-1):
        g[i+1] = g[i] + dt * (f[i+1]+f[i]) / 2
    return g

def uconv(dt, f, h):
    n = len(f) 
    g = np.empty(n)
    h = np.flip(h)
    g[0] = 0
    for i in np.arange(1, n):
        g[i] = np.trapz(f[:i+1]*h[-(i+1):], dx=dt)
    return g

def convolve(u, tc, c, th, h): # WIP
#    co(t) = int_0^t du h(u) c(t-u) 
    co = np.zeros(len(u))
    h = np.interp(u, th, h, left=0, right=0)
    c = np.interp(u, tc, c, left=0, right=0)
    for k, t in enumerate(u):
        if k != 0:
            ct = np.interp(t-u, u, c, left=0, right=0)
            co[k] = np.trapz(h[:k]*ct[:k], u[:k])
    return co  


def K_flow_1d(dx, u):
    nc = len(u)-1
    # Calculate Kn
    Kn = np.zeros(nc)
    un = u[:-1]
    neg = np.where(un < 0)
    Kn[neg] = -un[neg]/dx
    # Calculate Kp
    Kp = np.zeros(nc)
    up = u[1:]
    pos = np.where(up > 0)
    Kp[pos] = up[pos]/dx     
    return Kp, Kn

def K_diff_1d(dx, D):
    Kn = D[:-1]/dx**2
    Kp = D[1:]/dx**2   
    return Kp, Kn

def K_flowdiff_1d(dx, u, D):
    Ku = K_flow_1d(dx, u)
    Kd = K_diff_1d(dx, D)
    Kp = Ku[0] + Kd[0]
    Kn = Ku[1] + Kd[1]
    return Kp, Kn

def conc_1d1c(t, Jp, Kp, Kn):
    """Concentration in a spatial 1-compartment model in 1D"""
    nt = len(Jp)
    nc = len(Kp)
    K = Kp + Kn
    C = np.zeros((nt,nc))
    for k in range(nt-1):
        dt = t[k+1] - t[k]
        # Initialise at current concentration
        C[k+1,:] = C[k,:]
        # Add influxes at the boundaries:
        C[k+1,0] += dt*(Jp[k+1]+Jp[k])/2
        # Remove outflux to the neigbours:
        C[k+1,:] -= dt*K*C[k,:]
        # Add influx from the neighbours:
        C[k+1,:-1] += dt*Kn[1:]*C[k,1:]
        C[k+1,1:] += dt*Kp[:-1]*C[k,:-1]
    return C


def dt_1d1c(Kp, Kn):
    """maximal time step"""
    K = Kp + Kn
    return np.amin(1/K)

def conc_1d2cfp(t, Jp1, Kp1, Kn1, K02, E21):
    """Concentration in a spatial 2-compartment filtration model in 1D (positive influx only)"""
    nt = len(Jp1)
    nc = len(Kp1)
    # K21 = E21*(Kp1 + Kn1 + K21)
    # K21(1-E21) = E21*(Kp1 + Kn1)
    K21 = (Kp1 + Kn1)*E21/(1-E21)
    K1 = Kp1 + Kn1 + K21
    K2 = K02
    C1 = np.zeros((nt,nc))
    C2 = np.zeros((nt,nc))
    for k in range(nt-1):
        dt = t[k+1]-t[k]
        # Initialise at current concentration
        C1[k+1,:] = C1[k,:]
        C2[k+1,:] = C2[k,:]
        # Add influxes at the boundaries:
        C1[k+1,0] += dt*Jp1[k]
        # Remove outflux to the neigbours:
        C1[k+1,:] -= dt*K1*C1[k,:]
        C2[k+1,:] -= dt*K2*C2[k,:]
        # Add influx from the neighbours:
        C1[k+1,:-1] += dt*Kn1[1:]*C1[k,1:]
        C1[k+1,1:] += dt*Kp1[:-1]*C1[k,:-1]
        # Add influx at the same location
        C2[k+1,:] += dt*K21*C1[k,:]
    return C1, C2


def dt_1d2cfp(Kp1, Kn1, K02, E21):
    """maximal time step"""
    # K21 = E21*(Kp1 + Kn1 + K21)
    # K21(1-E21) = E21*(Kp1 + Kn1)
    K21 = (Kp1 + Kn1)*E21/(1-E21)
    K1 = Kp1 + Kn1 + K21
    K2 = K02
    # dt*K1<1
    # dt*K2<1
    return np.amin([np.amin(1/K1), np.amin(1/K2)])


def conc_1d3cf(t, Jp1, Kp1, K32, E21, Kn3):
    """Concentration in a spatial 3-compartment filtration model in 1D (positive influx only)"""
    nt = len(Jp1)
    nc = len(Kp1)
    # K21 = E21*(Kp1 + Kn1 + K21)
    # K21(1-E21) = E21*(Kp1 + Kn1)
    K21 = Kp1*E21/(1-E21)
    K1 = Kp1 + K21
    K2 = K32
    K3 = Kn3
    C1 = np.zeros((nt,nc))
    C2 = np.zeros((nt,nc))
    C3 = np.zeros((nt,nc))
    for k in range(nt-1):
        dt = t[k+1]-t[k]
        # Initialise at current concentration
        C1[k+1,:] = C1[k,:]
        C2[k+1,:] = C2[k,:]
        C3[k+1,:] = C3[k,:]
        # Add influxes at the boundaries:
        C1[k+1,0] += dt*Jp1[k]
        # Remove outflux to the neigbours:
        C1[k+1,:] -= dt*K1*C1[k,:]
        C2[k+1,:] -= dt*K2*C2[k,:]
        C3[k+1,:] -= dt*K3*C3[k,:]
        # Add influx from the neighbours:
        C1[k+1,1:] += dt*Kp1[:-1]*C1[k,:-1]
        C3[k+1,:-1] += dt*Kn3[1:]*C3[k,1:]
        # Add influx at the same location
        C2[k+1,:] += dt*K21*C1[k,:]
        C3[k+1,:] += dt*K32*C2[k,:]
    return C1, C2, C3

def dt_1d3cf(Kp1, K32, E21, Kn3):
    """maximal time step"""
    # K21 = E21*(Kp1 + Kn1 + K21)
    # K21(1-E21) = E21*(Kp1 + Kn1)
    K21 = Kp1*E21/(1-E21)
    K1 = Kp1 + K21
    K2 = K32
    K3 = Kn3
    # dt*K1<1
    # dt*K2<1
    return np.amin([np.amin(1/K1), np.amin(1/K2), np.amin(1/K3)])


def conc_1d2cxp(t, Jp1, Kp1, Kn1, K12, E21):
    """Concentration in a spatial 2-compartment exchange model in 1D (positive influx only)"""
    nt = len(Jp1)
    nc = len(Kp1)
    # K21 = E21*(Kp1 + Kn1 + K21)
    # K21(1-E21) = E21*(Kp1 + Kn1)
    K21 = (Kp1 + Kn1)*E21/(1-E21)
    K1 = Kp1 + Kn1 + K21
    K2 = K12
    C1 = np.zeros((nt,nc))
    C2 = np.zeros((nt,nc))
    for k in range(nt-1):
        dt = t[k+1]-t[k]
        # Initialise at current concentration
        C1[k+1,:] = C1[k,:]
        C2[k+1,:] = C2[k,:]
        # Add influxes at the boundaries:
        C1[k+1,0] += dt*Jp1[k]
        # Remove outflux to the neigbours:
        C1[k+1,:] -= dt*K1*C1[k,:]
        C2[k+1,:] -= dt*K2*C2[k,:]
        # Add influx from the neighbours:
        C1[k+1,:-1] += dt*Kn1[1:]*C1[k,1:]
        C1[k+1,1:] += dt*Kp1[:-1]*C1[k,:-1]
        # Add influx at the same location
        C2[k+1,:] += dt*K21*C1[k,:]
        C1[k+1,:] += dt*K12*C2[k,:]
    return C1, C2


def dt_1d2cxp(Kp1, Kn1, K12, E21):
    """maximal time step"""
    # K21 = E21*(Kp1 + Kn1 + K21)
    # K21(1-E21) = E21*(Kp1 + Kn1)
    K21 = (Kp1 + Kn1)*E21/(1-E21)
    K1 = Kp1 + Kn1 + K21
    K2 = K12
    # dt*K1<1
    # dt*K2<1
    return np.amin([np.amin(1/K1), np.amin(1/K2)])


def res_liver_sandwich(
            # Liver model 
            t, # time points (sec)
            J, # influx per unit volume (mmol/sec/mL tissue)
            vel, # velocity (1/sec)
            El, # Extraction fraction
            Kbh, # Biliary excretion rate (sec)
            velb, # bile velocity (1/sec)
            n = 20, # nr of numerical voxels
            ):
    # dimensionless locations of numerical voxel boundaries
    x = np.linspace(0, 1, n+1)
    dx = x[1] 
    # interpolate parameters on tube 
    vel_x = uinterp(x, vel, pos=True)
    Kbh_x = uinterp(x, Kbh, pos=True)
    El_x = uinterp(x, El, pos=True)
    velb_x = uinterp(x, velb, pos=True)
    Kbh_x = (Kbh_x[1:]+Kbh_x[:-1])/2
    El_x = (El_x[1:]+El_x[:-1])/2
    # Compartmental rate constants
    Kp, _ = K_flow_1d(dx, vel_x) # 1/sec
    _, Kb_n = K_flow_1d(dx, -velb_x) # 1/sec
    # High resolution time points
    dth = 0.9*dt_1d3cf(Kp, Kbh_x, El_x, Kb_n)
    tacq = np.amax(t)
    if tacq/dth > 1e6:
        print('dth', dth, tacq/dth)
        print('vel', vel)
        print('vel_x', vel_x) 
        print('velb', velb)
        print('velb_x', velb_x) 
        print('Kbh', Kbh)
        print('Kbh_x', Kbh_x)
        print('El', El)
        print('El_x', El_x)
        print('Kp', Kp)
        print('Kbn', Kb_n)
        msg = 'Rate constants are too high for numerical computation. Consider reducing their upper bound.'
        raise ValueError(msg)
    nth = 1 + np.ceil(tacq/dth).astype(np.int32)
    th = np.linspace(0, tacq, nth)
    # Upsample influx
    Jh = np.interp(th, t, J/dx)
    # Calculate concentrations
    Ce, Ch, Cb = conc_1d3cf(t, Jh, Kp, Kbh_x, El_x, Kb_n)
    # Sum concentrations over all voxels
    Ce = dx*np.sum(Ce, axis=1)
    Ch = dx*np.sum(Ch, axis=1)
    Cb = dx*np.sum(Cb, axis=1)
    # Downsample concentrations to measured time resolution.
    Ce = np.interp(t, th, Ce)
    Ch = np.interp(t, th, Ch)
    Cb = np.interp(t, th, Cb)
    return Ce, Ch, Cb # mmol/mL tissue

def res_liver(
            # Liver model 
            t, # time points (sec)
            J, # influx per unit volume (mmol/sec/mL tissue)
            vel, # velocity (1/sec)
            dif, # diffusion rate (1/sec)
            Kbh, # Biliary excretion rate (sec)
            El, # Extraction fraction
            n = 20, # nr of numerical voxels
            ):
    # dimensionless locations of numerical voxel boundaries
    x = np.linspace(0, 1, n+1)
    dx = x[1] 
    # interpolate parameters on tube 
    vel_x = uinterp(x, vel, pos=True)
    dif_x = uinterp(x, dif, pos=True)
    Kbh_x = uinterp(x, Kbh, pos=True)
    El_x = uinterp(x, El, pos=True)
    Kbh_x = (Kbh_x[1:]+Kbh_x[:-1])/2
    El_x = (El_x[1:]+El_x[:-1])/2
    # Compartmental rate constants
    Kp, Kn = K_flowdiff_1d(dx, vel_x, dif_x) # 1/sec
    # High resolution time points
    #dth = 0.9/(2*dif/dx**2 + np.amax(vel_x)/dx)
    dth = 0.9*dt_1d2cfp(Kp, Kn, Kbh_x, El_x)
    tacq = np.amax(t)
    if tacq/dth > 1e6:
        print('dth', dth, tacq/dth)
        print('vel', vel)
        print('vel_x', uinterp(x, vel, pos=True))
        print('dif', dif)
        print('dif_x', dif_x)
        print('Ti', Kbh)
        print('Ti_x', Kbh_x)
        print('Ei', El)
        print('Ei_x', El_x)
        print('Kp', Kp)
        print('Kn', Kn)
        msg = 'Rate constants are too high for numerical computation. Consider reducing their upper bound.'
        raise ValueError(msg)
    nth = 1 + np.ceil(tacq/dth).astype(np.int32)
    th = np.linspace(0, tacq, nth)
    # Upsample influx
    Jh = np.interp(th, t, J/dx)
    # Calculate concentrations
    Ce, Ch = conc_1d2cfp(th, Jh, Kp, Kn, Kbh_x, El_x)
    # Sum concentrations over all voxels
    Ce = dx*np.sum(Ce, axis=1)
    Ch = dx*np.sum(Ch, axis=1)
    # Downsample concentrations to measured time resolution.
    Ce = np.interp(t, th, Ce)
    Ch = np.interp(t, th, Ch)
    return Ce, Ch # mmol/mL tissue


def conc_nephi(
            # Nephron model with linearly varying reabsorption
            t, # time points (sec)
            J, # influx per unit volume (mmol/sec/mL tissue)
            vel, # velocity (1/sec)
            dif, # diffusion rate (1/sec)
            reabs, # reabsorption rate
            Ti, # Interstitial MTT (sec)
            Ei, # Interstitial extraction fraction
            n = 20, # nr of numerical voxels
            ):
    # dimensionless locations of numerical voxel boundaries
    x = np.linspace(0, 1, n+1)
    dx = x[1] 
    # interpolate parameters on tube 
    vel_x = uinterp(x, vel, pos=True)
    dif_x = uinterp(x, dif, pos=True)
    reabs_x = uinterp(x, reabs, pos=True)
    Ti_x = uinterp(x, Ti, pos=True)
    Ei_x = uinterp(x, Ei, pos=True)
    Ti_x = (Ti_x[1:]+Ti_x[:-1])/2
    Ei_x = (Ei_x[1:]+Ei_x[:-1])/2
    # reabsorption correction
    vel_x = vel_x*np.exp(-utrapz(dx, reabs_x))
    # Compartmental rate constants
    Kp, Kn = K_flowdiff_1d(dx, vel_x, dif_x) # 1/sec
    # High resolution time points
    #dth = 0.9/(2*dif/dx**2 + np.amax(vel_x)/dx)
    dth = 0.9*dt_1d2cxp(Kp, Kn, 1/Ti_x, Ei_x)
    tacq = np.amax(t)
    if tacq/dth > 1e6:
        print('dth', dth, tacq/dth)
        print('vel', vel)
        print('vel_x', uinterp(x, vel, pos=True))
        print('dif', dif)
        print('dif_x', dif_x)
        print('reabs', reabs)
        print('reabs_x', reabs_x)
        print('Ti', Ti)
        print('Ti_x', Ti_x)
        print('Ei', Ei)
        print('Ei_x', Ei_x)
        print('Kp', Kp)
        print('Kn', Kn)
        msg = 'Rate constants are too high for numerical computation. Consider reducing their upper bound.'
        raise ValueError(msg)
    nth = 1 + np.ceil(tacq/dth).astype(np.int32)
    th = np.linspace(0, tacq, nth)
    # Upsample influx
    Jh = np.interp(th, t, J/dx)
    # Calculate concentrations
    # Ch = conc_1d1c(th, Jh, Kp, Kn)
    Cht, Chi = conc_1d2cxp(th, Jh, Kp, Kn, 1/Ti_x, Ei_x)
    Ch = Cht + Chi
    # Sum concentrations over all voxels
    Cth = dx*np.sum(Ch, axis=1)
    # Downsample concentrations to measured time resolution.
    Ct = np.interp(t, th, Cth)
    return Ct # mmol/mL tissue


def conc_neph(
            # Nephron model with linearly varying reabsorption
            t, # time points (sec)
            J, # influx per unit volume (mmol/sec/mL tissue)
            vel, # velocity (1/sec)
            dif, # diffusion rate (1/sec)
            reabs, # reabsorption rate
            n = 20, # nr of numerical voxels
            ):
    # dimensionless locations of numerical voxel boundaries
    x = np.linspace(0, 1, n+1)
    dx = x[1] 
    # reabsorption rate everywhere
    reabs_x = uinterp(x, reabs, pos=True)
    # velocity everywhere
    vel_x = uinterp(x, vel, pos=True)
    vel_x = vel_x*np.exp(-utrapz(dx, reabs_x)) # reabs may be unnecessary
    # diffusion rate everywhere 
    dif_x = uinterp(x, dif, pos=True)
    # Compartmental rate constants
    Kp, Kn = K_flowdiff_1d(dx, vel_x, dif_x) # 1/sec
    # High resolution time points
    #dth = 0.9/(2*dif/dx**2 + np.amax(vel_x)/dx)
    dth = 0.9 * dt_1d1c(Kp, Kn)
    tacq = np.amax(t)
    if tacq/dth > 1e6:
        print('dth', dth, tacq/dth)
        print('vel', vel)
        print('vel_x', uinterp(x, vel, pos=True))
        print('dif', dif)
        print('dif_x', dif_x)
        print('reabs', reabs)
        print('reabs_x', reabs_x)
        print('Kp', Kp)
        print('Kn', Kn)
        msg = 'Tubular velocity is too high. Reduce upper bound on the velocity.'
        raise ValueError(msg)
    nth = 1 + np.ceil(tacq/dth).astype(np.int32)
    th = np.linspace(0, tacq, nth)
    # Upsample influx
    Jh = np.interp(th, t, J/dx)
    # Calculate concentrations
    Ch = conc_1d1c(th, Jh, Kp, Kn)
    # Sum concentrations over all voxels
    Cth = dx*np.sum(Ch, axis=1)
    # Downsample concentrations to measured time resolution
    Ct = np.interp(t, th, Cth)
    return Ct # mmol/mL tissue


def conc_plug_var(t, J, T:np.ndarray):
    nt = len(t)
    nx = len(T)
    C = np.zeros((nt,nx))
    Ji = J
    for i in range(nx):
        Jo = prop_plug(t, Ji, T[i])
        C[:,i] = trapz(t, Ji-Jo)
        Ji = Jo
    return C

def res_plug_var(t, J, T:np.ndarray):
    nx = len(T)
    Jx = J
    for i in range(nx):
        Jx = prop_plug(t, Jx, T[i])
    return trapz(t, J-Jx)


def conc_neph_plug(
            # Nephron model with linearly varying reabsorption
            t, # time points (sec)
            J, # influx per unit volume (mmol/sec/mL tissue)
            vel, 
            reabs, # MTT (sec)
            n = 20, # nr of numerical voxels
            ):
    x = np.linspace(0, 1, n+1)
    dx = x[1] 
    reabs_x = uinterp(x, reabs, floor=True)
    vel_x = uinterp(x, vel, floor=True)
    vel_x = vel_x*np.exp(-utrapz(dx, reabs_x)) 
    vel_x = (vel_x[1:]+vel_x[:-1])/2
    T_x = dx/vel_x
    Ct = res_plug_var(t, J, T_x)
    return Ct

def mtt_neph_plug(vel,reabs,n=20):
    x = np.linspace(0, 1, n+1)
    dx = x[1] 
    reabs_x = uinterp(x, reabs, floor=True)
    vel_x = uinterp(x, vel, floor=True)
    vel_x = vel_x*np.exp(-utrapz(dx, reabs_x)) 
    vel_x = (vel_x[1:]+vel_x[:-1])/2
    T_x = dx/vel_x
    return np.sum(T_x) 


def prop_plug(t, J, T):
    if T==0:
        return J
    return np.interp(t-T, t, J, left=0) 

def prop_comp(t, J, T):
    if T == np.inf:
        return np.zeros(len(t))
    return expconv(T, t, J)

def uprop_chain(dt, J, T, D):
    t = dt*np.arange(len(J))
    if D==0:
        return prop_plug(t,J,T)
    elif D==100:
        return prop_comp(t,J,T)
    else:
        h = chain_propagator(t, T, D)
        return uconv(dt, h, J)

# def uprop_chain(dt, J, T, D):
#     C = ures_chain(dt, J, T, D)
#     return J - np.gradient(C, dt)

def ures_chain(dt, J, T, D):
    t = dt*np.arange(len(J))
    R = chain_residue(t, T, D)
    return uconv(dt, R, J)

def res_plug(t, J, T):
    Jo = prop_plug(t, J, T)
    return trapz(t, J-Jo)

def res_trap(t, J):
    return trapz(t, J)

def prop_free(dt, tmax, J, H:np.ndarray, TT=None, TTmin=0, TTmax=None):
    if np.isscalar(H):
        H = [H]
    nTT = len(H)
    if TT is None:
        if TTmax is None:
            TTmax = tmax
        TT = np.linspace(TTmin, TTmax, nTT+1)
    else:
        if len(TT) != nTT+1:
            msg = 'The array of transit time boundaries needs to have length N+1, '
            msg += '\n with N the size of the transit time distribution H.'
            raise ValueError(msg)
    dist = rv_histogram((H,TT), density=True)
    t = np.linspace(0, tmax, len(J))
    H = dist.pdf(t)
    return uconv(dt, H, J)


def res_free(dt, tmax, J, H:np.ndarray, TT=None, TTmin=0, TTmax=None):
    if np.isscalar(H):
        H = [H]
    nTT = len(H)
    if TT is None:
        if TTmax is None:
            TTmax = tmax
        TT = np.linspace(TTmin, TTmax, nTT+1)
    else:
        if len(TT) != nTT+1:
            msg = 'The array of transit time boundaries needs to have length N+1, '
            msg += '\n with N the size of the transit time distribution H.'
            raise ValueError(msg)
    dist = rv_histogram((H,TT), density=True)
    t = np.linspace(0, tmax, len(J))
    R = 1 - dist.cdf(t)
    return uconv(dt, R, J)


def res_free_desc(tmax, H:np.ndarray, TT=None, TTmin=0, TTmax=None):
    if np.isscalar(H):
        H = [H]
    nTT = len(H)
    if TT is None:
        if TTmax is None:
            TTmax = tmax
        TT = np.linspace(TTmin, TTmax, nTT+1)
    else:
        if len(TT) != nTT+1:
            msg = 'The array of transit time boundaries needs to have length N+1, '
            msg += '\n with N the size of the transit time distribution H.'
            raise ValueError(msg)
    dist = rv_histogram((H,TT), density=True)
    return {
        'mean': dist.mean(),
        'median': dist.median(),
        'stdev': dist.std(),
        }


def res_comp(t, J, T):
    if T == np.inf:
        return trapz(t, J)
    return T*expconv(T, t, J)

def prop_2cfm(t, J, Ta, Tb, Eba):
    Ja = prop_comp(t, J, Ta)
    Jb = prop_comp(t, Eba*Ja, Tb)
    return (1-Eba)*Ja, Jb

def res_2cfm(t, J, Ta, Tb, Eba):
    Ja = prop_comp(t, J, Ta) 
    Jb = prop_comp(t, Eba*Ja, Tb)
    return Ta*Ja, Tb*Jb

def liver_2cfm_pars(fp, ve, kbh, khe, v=1):
    E = khe/(fp+khe)
    Te = ve/(fp+khe)
    vh = v-ve
    Th = vh/kbh
    return Te, Th, E

def liver_2cfm_invpars(ve, Te, E, Th, v=1):
    khe = E*ve/Te
    fp = (1-E)*ve/Te
    vh = v - ve
    kbh = vh/Th
    return fp, khe, vh, kbh
  
def compartment_propagator(t, MTT):
    return np.exp(-t/MTT)/MTT

def propagate_compartment(t, c, MTT):
    """Returns the average concentration at the outlet given the concentration at the inlet"""
    return expconv(MTT, t, c)

def residue_compartment(t, c, MTT):
    """Returns the concentration inside the system given the concentration at the inlet"""
    return propagate_compartment(t, c, MTT)

def propagate_dd(t, c, tdel, tdisp): 
    c = expconv(tdisp, t, c)
    if tdel != 0:
        c = np.interp(t-tdel, t, c, left=0)
    return c

def plug_residue(t, T):
    g = np.ones(len(t))
    g[np.where(t>T)] = 0
    return g

def comp_residue(t, T):
    return np.exp(-t/T)

def chain_residue(t, MTT, disp):
    if disp==0: # plug flow
        g = np.ones(len(t))
        return plug_residue(t, MTT)
    if disp==100: # compartment
        return comp_residue(t, MTT)
    n = 100/disp
    Tx = MTT/n
    norm = Tx*gamma(n)
    if norm == np.inf:
        return plug_residue(t, MTT)
    u = t/Tx  
    nt = len(t)
    g = np.ones(nt)
    g[0] = 0
    fnext = u[0]**(n-1)*np.exp(-u[0])/norm
    for i in range(nt-1):
        fi = fnext
        pow = u[i+1]**(n-1)
        if pow == np.inf:
            return 1-g
        fnext = pow * np.exp(-u[i+1])/norm
        g[i+1] = g[i] + (t[i+1]-t[i]) * (fnext+fi) / 2
    return 1-g

def res_plug_distr(t, J, MTT, disp, n=50, Tmax=5):
    # Define n+1 equally space transit times, including zero
    T = np.linspace(0, Tmax*MTT, n+1)
    # Determine the probabilities for each transit time
    H = chain_propagator(T, MTT, disp)
    # Determine the cumulative distribution of transit times
    CD = utrapz(T[1]-T[0], H)
    CD = (CD-np.amin(CD))/(np.amax(CD)-np.amin(CD))
    # Drop transit time zero and derive a probability for the others
    T = T[1:]
    H = CD[1:]-CD[:-1]
    # Add up the residue in each of the parallel paths
    N = np.zeros(len(t))
    for k in range(len(T)):
        N += res_plug(t, H[k]*J, T[k])
    return N


def chain_propagator(t, MTT, disp): # dispersion in %
    n = 100/disp
    Tx = MTT/n
    u = t/Tx
    return u**(n-1) * np.exp(-u)/Tx/gamma(n)

def residue_chain(t, ci, MTT, dispersion):
    co = propagate_chain(t, ci, MTT, dispersion)
    return np.trapz(ci-co, t)/MTT

def propagate_chain(t, ci, MTT, dispersion): # dispersion in % 

    if MTT == 0:
        return ci
    if dispersion == 0:
        return np.interp(t-MTT, t, ci, left=0)
    H = chain_propagator(t, MTT, dispersion)
    return convolve(t, t, ci, t, H)

def propagate_delay(t, c, delay):
    return np.interp(t-delay, t, c, left=0) 

def propagate_2cxm(t, ca, KP, KE, KB):
    """Calculate the propagators for the individual compartments in the 2CXM 
    
    For details and notations see appendix of 
    Sourbron et al. Magn Reson Med 62:672–681 (2009)

    Arguments
    ---------

    t : numpy array
        time points (sec) where the input function is defined
    ca : numpy array
        input function (mmol/mL)
    KP : float
        inverse plasma MTT (sec) = VP/(FP+PS)
    KE : float
        inverse extracellular MTT (sec) = VE/PS
    KB : float
        inverse blood MTT (sec) = VP/FP

    Returns
    -------
    cp : numpy array
        concentration in the plasma compartment (mmol/mL)
    ce : numpy array
        concentration in the extracellular compartment (mmol/mL)

    Examples
    --------
    coming soon..

    """

    KT = KP + KE
    sqrt = math.sqrt(KT**2-4*KE*KB)

    Kpos = 0.5*(KT + sqrt)
    Kneg = 0.5*(KT - sqrt)

    cpos = expconv(1/Kpos, t, ca)
    cneg = expconv(1/Kneg, t, ca)

    Eneg = (Kpos - KB)/(Kpos - Kneg)

    cp = (1-Eneg)*cpos + Eneg*cneg
    ce = (cneg*Kpos - cpos*Kneg) / (Kpos -  Kneg) 

    return cp, ce


def prop_2cxm(t, Ja, E, TP, TE):
    """Calculate the propagators for the individual compartments in the 2CXM 
    
    For details and notations see appendix of 
    Sourbron et al. Magn Reson Med 62:672–681 (2009)

    Arguments
    ---------

    t : numpy array
        time points (sec) where the input function is defined
    ca : numpy array
        input function (mmol/mL)
    KP : float
        (FP+PS)/VP
        inverse plasma MTT (sec) = VP/(FP+PS)
    KE : float
        inverse extracellular MTT (sec) = VE/PS
    KB : float
        FP/VP
        inverse blood MTT (sec) = VP/FP

    Returns
    -------
    cp : numpy array
        concentration in the plasma compartment (mmol/mL)
    ce : numpy array
        concentration in the extracellular compartment (mmol/mL)

    Examples
    --------
    coming soon..

    """
    # 1-E = KB/KP
    # KB = (1-E)KP
    KP = 1/TP
    KE = 1/TE
    KB = (1-E)*KP

    KT = KP + KE
    sqrt = math.sqrt(KT**2-4*KE*KB)

    Kpos = 0.5*(KT + sqrt)
    Kneg = 0.5*(KT - sqrt)

    Jpos = expconv(1/Kpos, t, Ja)
    Jneg = expconv(1/Kneg, t, Ja)

    Eneg = (Kpos - KB)/(Kpos - Kneg)

    Jp = (1-Eneg)*Jpos + Eneg*Jneg

    return Jp


def whole_body_uptake(t, J_inj,
        CO, T_lh,
        E_organs, Tp_organs, Te_organs,
        T_gut, FF_liver, E_liver, T_liver,  
        FF_kidneys, E_kidneys, T_kidneys,
        tol=0.001):
    J_aorta = prop_body(t, J_inj, 
        T_lh,
        E_organs, Tp_organs, Te_organs,
        T_gut, FF_liver, E_liver, T_liver,  
        FF_kidneys, E_kidneys, T_kidneys, 
        tol=tol)
    N_liver = res_liver_2cum(t, FF_liver*J_aorta, T_gut, E_liver, T_liver)
    N_kidneys = res_kidneys_2cum(t, FF_kidneys*(1-FF_liver)*J_aorta, E_kidneys, T_kidneys)
    c_aorta = J_aorta/CO
    return N_liver, N_kidneys, c_aorta

def whole_body_kidney_uptake(t, J_inj, 
        T_lh,
        E_organs, Tp_organs, Te_organs,
        T_gut, FF_liver, E_liver, Ke_liver, Kh_liver,
        FF_kidneys, E_kidneys, K_kidneys,
        tol=0.001):
    J_aorta = prop_body(t, J_inj, 
        T_lh,
        E_organs, Tp_organs, Te_organs,
        T_gut, FF_liver, E_liver, Ke_liver,  
        FF_kidneys, E_kidneys, K_kidneys, 
        tol=tol)
    J_liver = prop_comp(t, FF_liver*J_aorta, T_gut)
    Ne_liver, Nh_liver = res_liver_2cfm_ns(t, J_liver, E_liver, Ke_liver, Kh_liver) 
    J_kidneys = FF_kidneys*J_aorta
    Np_kidneys, Nt_kidneys = res_kidneys_2cum(t, J_kidneys, E_kidneys, K_kidneys)
    return Ne_liver, Nh_liver, Np_kidneys, Nt_kidneys, J_aorta

def res_liver_2cfm_ns(t, J_in, E, Ke, Kh):
    Ne = res_nscomp(t, J_in, Ke)
    Nh = res_nscomp(t, E*Ke*Ne, Kh)
    return Ne, Nh

def res_kidneys_2cum(t, J_in, E, Kp):
    Np = res_nscomp(t, J_in, Kp)
    Nt = res_trap(t, E*Kp*Np)
    return Np, Nt

def prop_body(t, J_lungs,
        T_lh, 
        E_organs, Tp_organs, Te_organs,
        T_gut, FF_liver, E_liver, Ke_liver, 
        FF_kidneys, E_kidneys, K_kidneys,
        tol = 0.001):
    dose0 = np.trapz(J_lungs, t)
    dose = dose0
    min_dose = tol*dose0
    J_lungs_total = J_lungs
    while dose > min_dose:
        # Propagate through the lungs and heart
        J_aorta = prop_comp(t, J_lungs, T_lh)
        # Split into liver, kidneys and other organs
        J_liver = FF_liver*J_aorta
        J_kidneys = FF_kidneys*J_aorta
        J_organs = (1-FF_kidneys-FF_liver)*J_aorta
        # Propagate through liver, kidneys and other organs
        J_liver = prop_comp(t, J_liver, T_gut)
        J_liver = (1-E_liver)*prop_nscomp(t, J_liver, Ke_liver)
        J_kidneys = (1-E_kidneys)*prop_nscomp(t, J_kidneys, K_kidneys)
        J_organs = prop_2cxm(t, J_organs, E_organs, Tp_organs, Te_organs)
        # Add up outfluxes from liver, kidneys and other organs
        J_lungs = J_liver + J_kidneys + J_organs
        # Add to the total flux into the lungs
        J_lungs_total += J_lungs
        # Get residual dose
        dose = np.trapz(J_lungs, t)
    # Propagate total flux through lungs
    J_aorta_total = prop_comp(t, J_lungs_total, T_lh)
    # Return total flux into aorta
    return J_aorta_total


def prop_body_kidneys(t, J_lungs,
        T_lh, 
        E_organs, Tp_organs, Te_organs,
        E_liver,
        FF_kidneys, E_kidneys, Ke_kidneys, 
        tol = 0.001):
    dose0 = np.trapz(J_lungs, t)
    dose = dose0
    min_dose = tol*dose0
    J_lungs_total = J_lungs
    while dose > min_dose:
        # Propagate through the lungs and heart
        J_aorta = prop_comp(t, J_lungs, T_lh)
        # Split into liver and other organs
        J_kidneys = FF_kidneys*J_aorta
        J_organs = (1-FF_kidneys)*J_aorta
        # Propagate through liver and other organs
        J_kidneys = (1-E_kidneys)*prop_nscomp(t, J_kidneys, Ke_kidneys)
        J_organs = (1-E_liver)*prop_2cxm(t, J_organs, E_organs, Tp_organs, Te_organs)
        # Add up outfluxes from liver and other organs
        J_lungs = J_kidneys + J_organs
        # Add to the total flux into the lungs
        J_lungs_total += J_lungs
        # Get residual dose
        dose = np.trapz(J_lungs, t)
    # Propagate total flux through lungs
    J_aorta_total = prop_comp(t, J_lungs_total, T_lh)
    # Return total flux into aorta
    return J_aorta_total

def prop_body2_kidneys(t, J_lungs,
        T_l, T_h,
        E_organs, Tp_organs, Te_organs,
        E_liver,
        FF_kidneys, E_kidneys, Ke_kidneys, 
        T_venous,
        tol = 0.001):
    dose0 = np.trapz(J_lungs, t)
    dose = dose0
    min_dose = tol*dose0
    J_lungs_total = J_lungs
    while dose > min_dose:
        # Propagate through the lungs and heart
        J_aorta = prop_comp(t, J_lungs, T_l)
        J_aorta = prop_comp(t, J_aorta, T_h)
        # Split into liver and other organs
        J_kidneys = FF_kidneys*J_aorta
        J_organs = (1-FF_kidneys)*J_aorta
        # Propagate through liver and other organs
        J_kidneys = (1-E_kidneys)*prop_nscomp(t, J_kidneys, Ke_kidneys)
        J_organs = (1-E_liver)*prop_2cxm(t, J_organs, E_organs, Tp_organs, Te_organs)
        # Add up outfluxes from liver and other organs
        J_lungs = J_kidneys + J_organs
        # Propgate through venous return
        J_lungs = prop_plug(t, J_lungs, T_venous)
        # Add to the total flux into the lungs
        J_lungs_total += J_lungs
        # Get residual dose
        dose = np.trapz(J_lungs, t)
    # Propagate total flux through lungs
    J_aorta_total = prop_comp(t, J_lungs_total, T_l)
    J_aorta_total = prop_comp(t, J_aorta_total, T_h)
    # Return total flux into aorta
    return J_aorta_total

def prop_body3_kidneys(t, J_lungs,
        T_lh, D_lh,
        E_organs, Tp_organs, Te_organs,
        E_liver,
        FF_kidneys, E_kidneys, Ke_kidneys, Ta_kidneys,
        tol = 0.001):
    dose0 = np.trapz(J_lungs, t)
    dose = dose0
    min_dose = tol*dose0
    J_lungs_total = J_lungs
    while dose > min_dose:
        # Propagate through the lungs and heart
        J_aorta = uprop_chain(t[1]-t[0], J_lungs, T_lh, D_lh)
        # Split into liver and other organs
        J_kidneys = FF_kidneys*J_aorta
        J_organs = (1-FF_kidneys)*J_aorta
        # Propagate through liver and other organs
        J_kidneys = prop_plug(t, J_kidneys, Ta_kidneys)
        J_kidneys = (1-E_kidneys)*prop_nscomp(t, J_kidneys, Ke_kidneys)
        J_organs = (1-E_liver)*prop_2cxm(t, J_organs, E_organs, Tp_organs, Te_organs)
        # Add up outfluxes from liver and other organs
        J_lungs = J_kidneys + J_organs
        # Add to the total flux into the lungs
        J_lungs_total += J_lungs
        # Get residual dose
        dose = np.trapz(J_lungs, t)
    # Propagate total flux through lungs
    J_aorta_total = uprop_chain(t[1]-t[0], J_lungs_total, T_lh, D_lh)
    # Return total flux into aorta
    return J_aorta_total


def prop_body_liver(t, J_lungs,
        T_lh, 
        E_organs, Tp_organs, Te_organs,
        T_gut, FF_liver, E_liver, Ke_liver, 
        E_kidneys,
        tol = 0.001):
    dose0 = np.trapz(J_lungs, t)
    dose = dose0
    min_dose = tol*dose0
    J_lungs_total = J_lungs
    while dose > min_dose:
        # Propagate through the lungs and heart
        J_aorta = prop_comp(t, J_lungs, T_lh)
        # Split into liver and other organs
        J_liver = FF_liver*J_aorta
        J_organs = (1-FF_liver)*J_aorta
        # Propagate through liver and other organs
        J_liver = prop_comp(t, J_liver, T_gut)
        J_liver = (1-E_liver)*prop_nscomp(t, J_liver, Ke_liver)
        J_organs = (1-E_kidneys)*prop_2cxm(t, J_organs, E_organs, Tp_organs, Te_organs)
        # Add up outfluxes from liver and other organs
        J_lungs = J_liver + J_organs
        # Add to the total flux into the lungs
        J_lungs_total += J_lungs
        # Get residual dose
        dose = np.trapz(J_lungs, t)
    # Propagate total flux through lungs
    J_aorta_total = prop_comp(t, J_lungs_total, T_lh)
    # Return total flux into aorta
    return J_aorta_total

def prop_body2_liver(t, J_lungs,
        T_lh, D_lh,
        E_organs, Tp_organs, Te_organs,
        T_gut, FF_liver, E_liver, Te_liver, 
        E_kidneys,
        tol = 0.001):
    dose0 = np.trapz(J_lungs, t)
    dose = dose0
    min_dose = tol*dose0
    J_lungs_total = J_lungs
    while dose > min_dose:
        # Propagate through the lungs and heart
        J_aorta = uprop_chain(t[1]-t[0], J_lungs, T_lh, D_lh)
        # Split into liver and other organs
        J_liver = FF_liver*J_aorta
        J_organs = (1-FF_liver)*J_aorta
        # Propagate through liver and other organs
        J_liver = prop_comp(t, J_liver, T_gut)
        J_liver = (1-E_liver)*prop_plug(t, J_liver, Te_liver)
        J_organs = (1-E_kidneys)*prop_2cxm(t, J_organs, E_organs, Tp_organs, Te_organs)
        # Add up outfluxes from liver and other organs
        J_lungs = J_liver + J_organs
        # Add to the total flux into the lungs
        J_lungs_total += J_lungs
        # Get residual dose
        dose = np.trapz(J_lungs, t)
    # Propagate total flux through lungs
    J_aorta_total = uprop_chain(t[1]-t[0], J_lungs_total, T_lh, D_lh)
    # Return total flux into aorta
    return J_aorta_total


def prop_body_liver_kidneys(t, J_lungs,
        T_lh, 
        E_organs, Tp_organs, Te_organs,
        T_gut, FF_liver, E_liver, Ke_liver, 
        FF_kidneys, E_kidneys, Kp_kidneys,
        tol = 0.001):
    dose0 = np.trapz(J_lungs, t)
    dose = dose0
    min_dose = tol*dose0
    J_lungs_total = J_lungs
    while dose > min_dose:
        # Propagate through the lungs and heart
        J_aorta = prop_comp(t, J_lungs, T_lh)
        # Split into liver, kidneys and other organs
        J_liver = FF_liver*J_aorta
        J_kidneys = FF_kidneys*J_aorta
        J_organs = (1-FF_liver-FF_kidneys)*J_aorta
        # Propagate through liver, kidneys and other organs
        J_liver = prop_comp(t, J_liver, T_gut)
        J_liver = (1-E_liver)*prop_nscomp(t, J_liver, Ke_liver)
        J_kidneys = (1-E_kidneys)*prop_nscomp(t, J_kidneys, Kp_kidneys)
        J_organs = prop_2cxm(t, J_organs, E_organs, Tp_organs, Te_organs)
        # Add up outfluxes from liver, kidneys and other organs
        J_lungs = J_liver + J_kidneys + J_organs
        # Add to the total flux into the lungs
        J_lungs_total += J_lungs
        # Get residual dose
        dose = np.trapz(J_lungs, t)
    # Propagate total flux through lungs
    J_aorta_total = prop_comp(t, J_lungs_total, T_lh)
    # Return total flux into aorta
    return J_aorta_total

def prop_body2_liver_kidneys(t, J_lungs,
        T_lh, D_lh,
        E_organs, Tp_organs, Te_organs,
        T_gut, FF_liver, E_liver, Ke_liver, 
        FF_kidneys, E_kidneys, Kp_kidneys,
        tol = 0.001):
    dose0 = np.trapz(J_lungs, t)
    dose = dose0
    min_dose = tol*dose0
    J_lungs_total = J_lungs
    while dose > min_dose:
        # Propagate through the lungs and heart
        J_aorta = uprop_chain(t[1]-t[0], J_lungs, T_lh, D_lh)
        # Split into liver, kidneys and other organs
        J_liver = FF_liver*J_aorta
        J_kidneys = FF_kidneys*J_aorta
        J_organs = (1-FF_liver-FF_kidneys)*J_aorta
        # Propagate through liver, kidneys and other organs
        J_liver = prop_comp(t, J_liver, T_gut)
        J_liver = (1-E_liver)*prop_nscomp(t, J_liver, Ke_liver)
        J_kidneys = (1-E_kidneys)*prop_nscomp(t, J_kidneys, Kp_kidneys)
        J_organs = prop_2cxm(t, J_organs, E_organs, Tp_organs, Te_organs)
        # Add up outfluxes from liver, kidneys and other organs
        J_lungs = J_liver + J_kidneys + J_organs
        # Add to the total flux into the lungs
        J_lungs_total += J_lungs
        # Get residual dose
        dose = np.trapz(J_lungs, t)
    # Propagate total flux through lungs
    J_aorta_total = uprop_chain(t[1]-t[0], J_lungs_total, T_lh, D_lh)
    # Return total flux into aorta
    return J_aorta_total


def res_liver_2cfm(t, J_in, T_gut, E, T_e, T_h):
    # Propagate through gut
    J_in = prop_plug(t, J_in, T_gut)
    # Extracellular outflux
    J_e = prop_comp(t, J_in, T_e)
    # Total concentration
    return T_e*J_e, res_comp(t, E*J_e, T_h)

def res_liver_2cum(t, J_in, T_gut, E, T_e):
    # Propagate through gut
    J_in = prop_plug(t, J_in, T_gut)
    # Extracellular outflux
    J_e = prop_comp(t, J_in, T_e)
    # Total concentration
    return T_e*J_e + res_trap(t, E*J_e)


def prop_simple_body(t, J_lungs,
        T_lh, 
        E_organs, Tp_organs, Te_organs,
        E_extr,
        tol = 0.001):
    dose0 = np.trapz(J_lungs, t)
    dose = dose0
    min_dose = tol*dose0
    J_lungs_total = J_lungs
    while dose > min_dose:
        # Propagate through the lungs and heart
        J_aorta = prop_comp(t, J_lungs, T_lh)
        # Propagate through the other organs
        J_lungs = (1-E_extr)*prop_2cxm(t, J_aorta, E_organs, Tp_organs, Te_organs)
        # Add to the total flux into the lungs
        J_lungs_total += J_lungs
        # Get residual dose
        dose = np.trapz(J_lungs, t)
    # Propagate total flux through lungs
    J_aorta_total = prop_comp(t, J_lungs_total, T_lh)
    # Return total flux into aorta
    return J_aorta_total

def prop_simple2_body(t, J_lungs,
        T_l, T_h, 
        E_organs, Tp_organs, Te_organs,
        E_extr, T_v,
        tol = 0.001):
    dose0 = np.trapz(J_lungs, t)
    dose = dose0
    min_dose = tol*dose0
    J_lungs_total = J_lungs
    while dose > min_dose:
        # Propagate through the lungs and heart
        J_aorta = prop_comp(t, J_lungs, T_l)
        J_aorta = prop_comp(t, J_aorta, T_h)
        # Propagate through the other organs
        J_lungs = (1-E_extr)*prop_2cxm(t, J_aorta, E_organs, Tp_organs, Te_organs)
        # Propgate through venous return
        J_lungs = prop_plug(t, J_lungs, T_v)
        # Add to the total flux into the lungs
        J_lungs_total += J_lungs
        # Get residual dose
        dose = np.trapz(J_lungs, t)
    # Propagate total flux through lungs
    J_aorta_total = prop_comp(t, J_lungs_total, T_l)
    J_aorta_total = prop_comp(t, J_aorta_total, T_h)
    # Return total flux into aorta
    return J_aorta_total


def prop_simple3_body(t, J_lungs,
        T_lh, D_lh,
        E_organs, Tp_organs, Te_organs,
        E_extr,
        tol = 0.001):
    dose0 = np.trapz(J_lungs, t)
    dose = dose0
    min_dose = tol*dose0
    J_lungs_total = J_lungs
    while dose > min_dose:
        # Propagate through the lungs and heart
        J_aorta = uprop_chain(t[1]-t[0], J_lungs, T_lh, D_lh)
        # Propagate through the other organs
        J_lungs = (1-E_extr)*prop_2cxm(t, J_aorta, E_organs, Tp_organs, Te_organs)
        # Add to the total flux into the lungs
        J_lungs_total += J_lungs
        # Get residual dose
        dose = np.trapz(J_lungs, t)
    # Propagate total flux through lungs
    J_aorta_total = uprop_chain(t[1]-t[0], J_lungs_total, T_lh, D_lh)
    # Return total flux into aorta
    return J_aorta_total




def Jin(t, vol, conc, rate, BAT):
    duration = vol/rate     # sec = mL / (mL/sec)
    Jmax = conc*rate        # mmol/sec = (mmol/ml) * (ml/sec)
    J = np.zeros(t.size)
    t_inject = (t > BAT) & (t < BAT+duration)
    J[np.nonzero(t_inject)[0]] = Jmax
    #J[t_inject] = Jmax # This should work too
    return J


def propagate_simple_body(t, c_vena_cava, 
    MTTlh, Eint, MTTe, MTTo, TTDo, Eext, tol=0.001):
    """Propagation through a 2-site model of the body."""

    dose0 = np.trapz(c_vena_cava, t)
    dose = dose0
    min_dose = tol*dose0

    c_vena_cava_total = 0*t
    c_aorta_total = 0*t

    while dose > min_dose:
        c_aorta = expconv(MTTlh, t, c_vena_cava)
        c_aorta_total += c_aorta
        c_vena_cava_total += c_vena_cava
        c = propagate_dd(t, c_aorta, MTTo, TTDo)
        c = (1-Eint)*c + Eint*expconv(MTTe, t, c) 
        c_vena_cava = c*(1-Eext)
        dose = np.trapz(c_vena_cava, t)

    return c_vena_cava_total, c_aorta_total


def residue_high_flow_ccf(t, ci, Ktrans, Te, De, FiTi):
    """Residue for a compartment i with high flow (Ti=0) and a chain e"""

    ni = FiTi*ci
    ne = (Te*Ktrans)*residue_chain(t, ci, Te, De)
    return ni, ne

def residue_high_flow_2cfm(t, ci, Ktrans, Te, FiTi):
    """Central compartment i with high flow (Ti=0) and filtration compartment e"""

    ni = FiTi*ci
    ne = (Te*Ktrans)*propagate_compartment(t, ci, Te)
    return ni, ne



def res_nscomp(t, J, K, C0=0):
    if not isinstance(K, np.ndarray):
        return res_comp(t, J, 1/K)
    #dtK must be >= 0 everywhere
    #dC(t)/dt = -K(t)C(t) + J(t) #mmol/mL/sec /sec mmol/ mmol/sec
    #C(t+dt)-C(t) = -dtK(t)C(t) + dtJ(t)
    #C(t+dt) = C(t) - dtK(t)C(t) + dtJ(t)
    #C(t+dt) = (1-dtK(t))C(t) + dtJ(t)
    n = len(t)
    C = np.zeros(n)
    C[0] = C0
    for i in range(n-1):
        dt = t[i+1]-t[i]
        R = 1-dt*K[i]
        C[i+1] = R*C[i] + dt*J[i] 
    return C

def prop_nscomp(t, J, K, C0=0):
    C = res_nscomp(t, J, K, C0=C0)
    return K*C

def residue_high_flow_2cfm_varK(t, ci, Ktrans1, Ktrans2, Ktrans3, Te, FiTi):
    """Central compartment i with high flow (Ti=0) and filtration compartment e"""

    mid = math.floor(len(t)/2)
    Ktrans = quadratic(t, t[0], t[mid], t[-1], Ktrans1, Ktrans2, Ktrans3)

    dt = t[1]-t[0]
    nt = len(t)

    ni = FiTi*ci
    ji = dt*Ktrans*ci
    Re = 1-dt/Te
    ne = np.empty(nt)
    ne[0] = 0
    for k in range(nt-1):
        ne[k+1] = ji[k] + Re * ne[k]
    return ni, ne, Ktrans

def residue_high_flow_2cfm_varT(t, ci, Ktrans, Te1, Te2, Te3, FiTi):
    """Central compartment i with high flow (Ti=0) and filtration compartment e"""

    # ve dce/dt = Ktrans*ci - k*ce
    # dne/dt = Ktrans * ci - ne / Te
    # Analytical solution with constant Te:
    #   ne(t) = exp(-t/Te) * Ktrans ci(t) 
    #   ne(t) = Te Ktrans P(Te, t) * ci(t)
    # Numerical solution with variable Te
    #   (ne(t+dt)-ne(t))/dt = Ktrans ci(t) - ne(t) / Te(t)
    #   ne(t+dt) = ne(t) + dt Ktrans ci(t) - ne(t) dt/Te(t)
    #   ne(t+dt) = dt Ktrans ci(t) + (1-dt/Te(t)) * ne(t)

    # Build time-varying Te (step function)
    # Te = np.empty(len(t))
    # tmid = math.floor(len(t)/2)
    # Te[:tmid] = Te1
    # Te[tmid:] = Te2
    mid = math.floor(len(t)/2)
    Te = quadratic(t, t[0], t[mid], t[-1], Te1, Te2, Te3)

    dt = t[1]-t[0]
    nt = len(t)

    ni = FiTi*ci
    ji = dt*Ktrans*ci
    Re = 1-dt/Te
    ne = np.empty(nt)
    ne[0] = 0
    for k in range(nt-1):
        ne[k+1] = ji[k] + Re[k] * ne[k]
    return ni, ne


def residue_high_flow_2cfm_varlinT(t, ci, Ktrans, Te1, Te2, FiTi):
    """Central compartment i with high flow (Ti=0) and filtration compartment e"""

    # ve dce/dt = Ktrans*ci - k*ce
    # dne/dt = Ktrans * ci - ne / Te
    # Analytical solution with constant Te:
    #   ne(t) = exp(-t/Te) * Ktrans ci(t) 
    #   ne(t) = Te Ktrans P(Te, t) * ci(t)
    # Numerical solution with variable Te
    #   (ne(t+dt)-ne(t))/dt = Ktrans ci(t) - ne(t) / Te(t)
    #   ne(t+dt) = ne(t) + dt Ktrans ci(t) - ne(t) dt/Te(t)
    #   ne(t+dt) = dt Ktrans ci(t) + (1-dt/Te(t)) * ne(t)

    # Build time-varying Te (step function)
    # Te = np.empty(len(t))
    # tmid = math.floor(len(t)/2)
    # Te[:tmid] = Te1
    # Te[tmid:] = Te2
    Te = linear(t, t[0], t[-1], Te1, Te2)

    dt = t[1]-t[0]
    nt = len(t)

    ni = FiTi*ci
    ji = dt*Ktrans*ci
    Re = 1-dt/Te
    ne = np.empty(nt)
    ne[0] = 0
    for k in range(nt-1):
        ne[k+1] = ji[k] + Re[k] * ne[k]
    return ni, ne


def injection(t, weight, conc, dose1, rate, start1, dose2=None, start2=None):
    """dose injected per unit time (mM/sec)"""

    duration = weight*dose1/rate     # sec = kg * (mL/kg) / (mL/sec)
    Jmax = conc*rate                # mmol/sec = (mmol/ml) * (ml/sec)
    t_inject = (t > 0) & (t < duration)
    J = np.zeros(t.size)
    J[np.nonzero(t_inject)[0]] = Jmax
    J1 = propagate_delay(t, J, start1)
    if start2 is None:
        return J1
    duration = weight*dose2/rate     # sec = kg * (mL/kg) / (mL/sec)
    Jmax = conc*rate                # mmol/sec = (mmol/ml) * (ml/sec)
    t_inject = (t > 0) & (t < duration)
    J = np.zeros(t.size)
    J[np.nonzero(t_inject)[0]] = Jmax
    J2 = propagate_delay(t, J, start2)
    return J1 + J2

def injection_gv(t, weight, conc, dose, rate, start1, start2=None, dispersion=0.5):
    """dose injected per unit time (mM/sec)"""

    duration = weight*dose/rate     # sec = kg * (mL/kg) / (mL/sec)
    amount = conc*weight*dose       # mmol = (mmol/ml) * kg * (ml/kg)
    J = amount * chain_propagator(t, duration, dispersion) # mmol/sec
    J1 = propagate_delay(t, J, start1)
    if start2 is None:
        return J1
    else:
        J2 = propagate_delay(t, J, start2)
        return J1 + J2

def signalSPGRESS(TR, FA, R1, S0):

    E = np.exp(-TR*R1)
    cFA = np.cos(FA*math.pi/180)
    return S0 * (1-E) / (1-cFA*E)

def signal_genflash(TR, R1, S0, a, A):
    """Steady-state model of a spoiled gradient echo but
    parametrised with cos(FA) instead of FA and generalised to include rate.
    0<S0
    0<a
    -1<A<+1
    """
    E = np.exp(-a*TR*R1)
    return S0 * (1-E) / (1-A*E)

def signal_hyper(TR, R1, S0, a, b):
    """
    Descriptive bi-exponentional model for SPGRESS sequence.

    S = S0 (e^(+ax) - e^(-bx)) / (e^(+ax) + e^(-bx))
    with x = TR*R1
    0 < S
    0 < a
    0 < b
    """
    x = TR*R1
    Ea = np.exp(+a*x)
    Eb = np.exp(-b*x)
    return S0 * (Ea-Eb)/(Ea+Eb)

def signalBiExp(TR, R1, S0, A, a, b):
    """
    Descriptive bi-exponentional model for SPGRESS sequence.

    S = S0 (1 - A e^(-ax) - (1-A) e^(-bx))
    with x = TR*R1
    0 < A < 1
    0 < S
    0 < a
    0 < b
    """
    x = TR*R1
    Ea = np.exp(-a*x)
    Eb = np.exp(-b*x)
    return S0 * (1 - A*Ea - (1-A)*Eb)

def uinterp(x,y, pos=False, floor=False):
    # Interpolate y on x, assuming y-values are uniformly distributed over the x-range
    if np.isscalar(y):
        yi = y*np.ones(len(x))
    elif len(y)==1:
        yi = y[0]*np.ones(len(x))
    elif len(y)==2:
        yi = lin(x,y)
    elif len(y)==3:
        yi = quad(x,y)
    else:
        x_y = np.linspace(np.amin(x), np.amax(x), len(y))
        yi = CubicSpline(x_y, y)(x)
    if pos:
        yi[yi<0] = 0
    if floor:
        y0 = np.amin(y)
        yi[yi<y0] = y0
    return yi

def quad(t, K):
    nt = len(t)
    mid = math.floor(nt/2)
    return quadratic(t, t[0], t[mid], t[-1], K[0], K[1], K[2])

def lin(t, K):
    return linear(t, t[0], t[-1], K[0], K[1])

# def quadratic(x, x1, x2, x3, y1, y2, y3):
#     """returns a quadratic function of x 
#     that goes through the three points (xi, yi)"""

#     a = x1*(y3-y2) + x2*(y1-y3) + x3*(y2-y1)
#     a /= (x1-x2)*(x1-x3)*(x2-x3)
#     b = (y2-y1)/(x2-x1) - a*(x1+x2)
#     c = y1-a*x1**2-b*x1
#     return a*x**2+b*x+c

def linear(x, x1, x2, y1, y2):
    """returns a linear function of x 
    that goes through the two points (xi, yi)"""

    L1 = (x2-x)/(x2-x1)
    L2 = (x-x1)/(x2-x1)
    return y1*L1 + y2*L2

    b = (y2-y1)/(x2-x1)
    c = y1-b*x1
    return b*x+c

def quadratic(x, x0, x1, x2, y0, y1, y2):
    """returns a quadratic function of x 
    that goes through the three points (xi, yi)"""

    L0 = (x-x1)*(x-x2)/((x0-x1)*(x0-x2))
    L1 = (x-x0)*(x-x2)/((x1-x0)*(x1-x2))
    L2 = (x-x0)*(x-x1)/((x2-x0)*(x2-x1))
    return y0*L0 + y1*L1 + y2*L2



def concentrationSPGRESS(S, S0, T10, FA, TR, r1):
    """
    Calculates the tracer concentration from a spoiled gradient-echo signal.

    Arguments
    ---------
        S: Signal S(C) at concentration C
        S0: Precontrast signal S(C=0)
        FA: Flip angle in degrees
        TR: Repetition time TR in msec (=time between two pulses)
        T10: Precontrast T10 in msec
        r1: Relaxivity in Hz/mM

    Returns
    -------
        Concentration in mM
    """
    E = math.exp(-TR/T10)
    c = math.cos(FA*math.pi/180)
    Sn = (S/S0)*(1-E)/(1-c*E)	#normalized signal
    R1 = -np.log((1-Sn)/(1-c*Sn))/TR	#relaxation rate in 1/msec
    return 1000*(R1 - 1/T10)/r1 


def concentration_lin(S, S0, T10, r1):
    """
    Calculates the tracer concentration from a spoiled gradient-echo signal.

    Arguments
    ---------
        S: Signal S(C) at concentration C
        S0: Precontrast signal S(C=0)
        T10: Precontrast T10 in msec
        r1: Relaxivity in Hz/mM

    Returns
    -------
        Concentration in mM
    """
    R10 = 1/T10
    R1 = R10*S/S0	#relaxation rate in 1/msec
    return 1000*(R1 - R10)/r1 


def sample(t, S, ts, dts): 
    """Sample the signal assuming sample times are at the start of the acquisition"""

    Ss = np.empty(len(ts)) 
    for k, tk in enumerate(ts):
        tacq = (t >= tk) & (t < tk+dts)
        data = S[np.nonzero(tacq)[0]]
        Ss[k] = np.average(data)
    return Ss 


def test_quad():
    n = 50
    y = [0.009080495403170533, 0.00015172348866885195, 0.028591612238011686]
    x = np.linspace(0, 1, n+1)
    y_x = uinterp(x, y, pos=True)
    print(y_x)
    plt.plot(x,y_x)
    plt.show()

if __name__ == "__main__":
    test_quad()
