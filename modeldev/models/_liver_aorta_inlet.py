import os
import math
import numpy as np
import matplotlib.pyplot as plt

import dcmri
from curvefit import CurveFit


class OneShotOneScan(CurveFit):

    def parameters(self):

        return [
            ['S0', "Signal amplitude S0", 1200, "a.u.", 0, np.inf, False, 4],
            ['FA', "Flip angle", self.aorta.FA, "deg", 0, 180, False, 4],
            ['Ktrans', "Hepatocellular uptake rate", self.Ktrans, 'mL/sec/mL', 0, np.inf, True, 5],
            ['Th', "Hepatocellular mean transit time", 20*60., 'sec', 0, np.inf, True, 5],
            ['FpTe', "Apparent liver extracellular volume", self.veL, 'mL/mL', 0, 1, True, 3],
            ['tdel', "Gut delay time", 7.0, 'sec', 0, np.inf, False, 3],
            ['tdisp', "Gut dispersion time", 28.0, 'sec', 0, np.inf, False, 3],
        ]
    
    def R1(self):

        p = self.p.value
        cv = dcmri.propagate_dd(self.aorta.t, self.aorta.cb, p.tdel, p.tdisp)
        # ca = dcmri.propagate_delay(self.aorta.t, self.aorta.cb, p.MTTa)
        # self.cb = p.AFF*ca + (1-p.AFF)*cv
        self.cb = cv

        #cp = expconv(Te, t, cp)            # Dispersion in liver extracellular space

        cp = self.cb/self.aorta.Hct
        ne, nh = dcmri.residue_high_flow_2cfm(self.aorta.t, cp, p.Ktrans, p.Th, p.FpTe)

        self.ce = (1-p.Ktrans/self.Fp)*cp
        self.ch = nh/self.vh
        self.cl = ne + nh

        return self.R10 + self.aorta.rp*ne + self.rh*nh
    
    def function(self, x, p):

        R1 = self.R1()

        # Calculate signal and sample
        self.signal = dcmri.signalSPGRESS(self.aorta.TR, p.FA, R1, p.S0)
        return dcmri.sample(self.aorta.t, self.signal, x, self.aorta.tacq)

    def set_export_pars(self):
        p = self.p.value
        self.export_pars = self.p.drop(['initial value','lower bound','upper bound','fit','digits'], axis=1)
        self.export_pars.drop(['FA'],inplace=True)
        # Convert to conventional units
        self.export_pars.at['Ktrans','unit'] = 'mL/min/mL'
        self.export_pars.at['Ktrans','value'] = 60*p.Ktrans 
        self.export_pars.at['Th','unit'] = 'min'
        self.export_pars.at['Th','value'] = p.Th/60  
        # Add derived parameters   
        self.export_pars.loc['kbh'] = ["Biliary excretion rate", 60*self.vh/p.Th, 'mL/min/mL']

    def __init__(self, aorta):

        self.aorta = aorta
        self.aorta.signal_smooth()
        super().__init__()

    # Liver parameters
    Fb = 1.3/60         # Blood flow (mL/sec/mL)
                        # 130 mL/min/100mL  
    E = 0.04            # Gadoxetate extraction fraction
    veL = 0.230         # Liver extracellular volume (mL/mL)
    vh = 0.722          # Hepatocellular volume fraction (mL/mL)

    @property
    def R10lit(self):
        field = math.floor(self.aorta.field_strength)
        if field == 1.5: return 1000.0/602.0     # liver R1 in 1/sec (Waterton 2021)
        if field == 3.0: return 1000.0/752.0     # liver R1 in 1/sec (Waterton 2021)
        if field == 4.0: return 1.281     # liver R1 in 1/sec (Changed from 1.285 on 06/08/2020)
        if field == 7.0: return 1.109     # liver R1 in 1/sec (Changed from 0.8350 on 06/08/2020)
        if field == 9.0: return 0.920     # per sec - liver R1 (https://doi.org/10.1007/s10334-021-00928-x)


    def signal_smooth(self):

        R1 = self.R1()
        return dcmri.signalSPGRESS(self.aorta.TR, self.p.value.FA, R1, self.p.value.S0)

    @property
    def rh(self):
        field = math.floor(self.aorta.field_strength)
        if field == 1.5: return 14.6     # relaxivity of hepatocytes in Hz/mM
        if field == 3.0: return 9.8     # relaxivity of hepatocytes in Hz/mM
        if field == 4.0: return 7.6     # relaxivity of hepatocytes in Hz/mM
        if field == 7.0: return 6.0     # relaxivity of hepatocytes in Hz/mM
        if field == 9.0: return 6.1     # relaxivity of hepatocytes in Hz/mM

    @property
    def Fp(self):
        return (1-self.aorta.Hct) * self.Fb

    @property
    def khe(self):
        return self.E*self.Fp/(1-self.E)

    @property
    def Ktrans(self):
        return self.E * self.Fp 

    @property
    def Te(self):
        return self.veL*(1-self.E)/self.Fp

    @property
    def ve_app(self):
        return (1-self.E)*self.veL

    def set_R10(self, t, R1):
        self.R10 = R1

    def set_R11(self, t, R1):
        self.R11 = [t, R1]

    def estimate_p(self):

        baseline = np.nonzero(self.x <= self.aorta.p.value.BAT-5)[0]
        n0 = baseline.size
        if n0 == 0: 
            n0 = 1
        Sref = dcmri.signalSPGRESS(self.aorta.TR, self.p.value.FA, self.R10, 1)
        S0 = np.mean(self.y[:n0]) / Sref
        self.p.at['S0','value'] = S0

    def plot_fit(self, show=True, save=False, path=None, prefix=''):

        self.plot_with_conc(show=show, save=save, path=path, prefix=prefix)
        BAT = self.aorta.p.value.BAT
        self.plot_with_conc(win='pass1', xrange=[BAT-20, BAT+160], show=show, save=save, path=path, prefix=prefix)
        self.plot_with_conc(win='pass1_', xrange=[BAT-20, BAT+600], show=show, save=save, path=path, prefix=prefix)
        self.plot_with_conc(win='pass1__', xrange=[BAT-20, BAT+1200], show=show, save=save, path=path, prefix=prefix)

    def plot_with_conc(self, fit=True, xrange=None, win='all', show=True, save=False, path=None, legend=True, prefix=''):

        if fit is True:
            y = self.y
        else:
            y = self.yp
        if xrange is None:
            t0 = self.aorta.t[0]
            t1 = self.aorta.t[-1]
        else:
            t0 = xrange[0]
            t1 = xrange[1]
        if path is None:
            path = self.path()
        if not os.path.isdir(path):
            os.makedirs(path)
        ti = np.nonzero((self.aorta.t>=t0) & (self.aorta.t<=t1))[0]
        xi = np.nonzero((self.x>=t0) & (self.x<=t1))[0]

        fig, (ax1, ax2) = plt.subplots(1,2,figsize=(20,8))
        fig.suptitle("model fit " + win)

        ax1.set_title('Signal')
        ax1.set(xlabel=self.xlabel, ylabel=self.ylabel)
        ax1.plot(self.x[xi]+self.aorta.tacq/2, y[xi], 'ro', label='data')
        ax1.plot(self.aorta.t[ti], self.signal_smooth()[ti], 'b-', label=self.plabel())
        if (self.R11[0] >=t0) & (self.R11[0] <=t1):
            testsignal = dcmri.signalSPGRESS(self.aorta.TR, self.p.value.FA, self.R11[1], self.p.value.S0)
            ax1.plot(self.R11[0], testsignal, 'gx', label='test data (MOLLI)')
        if legend:
            ax1.legend()

        ax2.set_title('Reconstructed concentration')
        ax2.set(xlabel=self.xlabel, ylabel='Concentration (mM)')
        ax2.plot(self.aorta.t[ti], 0*self.aorta.t[ti], color='black')
        ax2.plot(self.aorta.t[ti], self.cl[ti], 'b-', label='liver')
        ax2.plot(self.aorta.t[ti], self.ce[ti], 'r-', label='extracellular')
        ax2.plot(self.aorta.t[ti], self.ch[ti], 'g-', label='hepatocyte')
        if legend:
            ax2.legend()
        if save:    
            if not os.path.exists(path):
                os.makedirs(path)      
            plt.savefig(fname=os.path.join(path, prefix + '_' + win + '.png'))
        if show:
            plt.show()
        else:
            plt.close()


class TwoShotTwoScan(OneShotOneScan):

    def function(self, x, p):

        R1 = self.R1()
        k2 = np.nonzero(self.aorta.t >= self.aorta.x2[0])[0]

        # Signals
        self.signal = dcmri.signalSPGRESS(self.aorta.TR, p.FA1, R1, p.S01)
        self.signal[k2] = dcmri.signalSPGRESS(self.aorta.TR, p.FA2, R1[k2], p.S02)
        return dcmri.sample(self.aorta.t, self.signal, x, self.aorta.tacq)

    def parameters(self):

        return [
            ['S01', "Signal amplitude S0", 1200, "a.u.", 0, np.inf, False, 4],
            ['FA1', "Flip angle 1", self.aorta.FA, "deg", 0, 180, False, 4],
            ['S02', "Signal amplitude S0", 1200, "a.u.", 0, np.inf, True, 4],
            ['FA2', "Flip angle 2", self.aorta.FA, "deg", 0, 180, False, 4],
            ['Ktrans', "Hepatocellular uptake rate", self.Ktrans, 'mL/sec/mL', 0, np.inf, True, 5],
            ['Th', "Hepatocellular mean transit time", 20*60., 'sec', 0, np.inf, True, 5],
            ['FpTe', "Apparent liver extracellular volume", self.veL, 'mL/mL', 0, 1, True, 3],
            ['tdel', "Gut delay time", 7.0, 'sec', 0, np.inf, False, 3],
            ['tdisp', "Gut dispersion time", 28.0, 'sec', 0, np.inf, False, 3],
        ]
    
    def set_export_pars(self):
        p = self.p.value
        self.export_pars = self.p.drop(['initial value','lower bound','upper bound','fit','digits'], axis=1)
        self.export_pars.drop(['FA1', 'FA2'],inplace=True)
        # Convert to conventional units
        self.export_pars.at['Ktrans','unit'] = 'mL/min/mL'
        self.export_pars.at['Ktrans','value'] = 60*p.Ktrans 
        self.export_pars.at['Th','unit'] = 'min'
        self.export_pars.at['Th','value'] = p.Th/60  
        # Add derived parameters   
        self.export_pars.loc['kbh'] = ["Biliary excretion rate", 60*self.vh/p.Th, 'mL/min/mL']

    def R1(self):

        p = self.p.value
        cv = dcmri.propagate_dd(self.aorta.t, self.aorta.cb, p.tdel, p.tdisp)
        # ca = dcmri.propagate_delay(self.aorta.t, self.aorta.cb, p.MTTa)
        # self.cb = p.AFF*ca + (1-p.AFF)*cv
        self.cb = cv

        #cp = expconv(Te, t, cp)            # Dispersion in liver extracellular space

        cp = self.cb/self.aorta.Hct
        ne, nh = dcmri.residue_high_flow_2cfm(self.aorta.t, cp, p.Ktrans, p.Th, p.FpTe)
        self.ce = (1-p.Ktrans/self.Fp)*cp
        self.ch = nh/self.vh
        self.cl = ne + nh

        # Relaxation rate
        return self.R10 + self.aorta.rp*ne + self.rh*nh

    def signal_smooth(self):

        R1 = self.R1()
        k2 = np.nonzero(self.aorta.t >= self.aorta.x2[0])[0]
        signal = dcmri.signalSPGRESS(self.aorta.TR, self.p.value.FA1, R1, self.p.value.S01)
        signal[k2] = dcmri.signalSPGRESS(self.aorta.TR, self.p.value.FA2, R1[k2], self.p.value.S02)
        return signal

    def estimate_p(self):

        BAT1 = self.aorta.p.value.BAT1
        Sref = dcmri.signalSPGRESS(self.aorta.TR, self.p.value.FA1, self.R10, 1)
        baseline = np.nonzero(self.x1 <= BAT1-20)[0]
        n0 = baseline.size
        if n0 == 0: 
            n0 = 1
        S01 = np.mean(self.y1[:n0]) / Sref
        self.p.at['S01','value'] = S01

        BAT2 = self.aorta.p.value.BAT2
        Sref = dcmri.signalSPGRESS(self.aorta.TR, self.p.value.FA2, self.R12[1], 1)
        n0 = math.floor(60/self.aorta.tacq) # 1 minute baseline
        S02 = np.mean(self.y2[:n0]) / Sref
        self.p.at['S02','value'] = S02


    def set_R12(self, t, R1):
        self.R12 = [t, R1]

    def set_x(self, x1, x2, **kwargs):
        self.x1 = x1
        self.x2 = x2
        super().set_x(np.append(x1, x2), **kwargs)

    def set_y(self, y1, y2):
        self.y = np.append(y1, y2)
        self.y1 = y1
        self.y2 = y2

    def plot_fit(self, show=True, save=False, path=None, prefix=''):

        self.plot_with_conc(show=show, save=save, path=path, prefix=prefix)
        BAT = self.aorta.p.value.BAT1
        self.plot_with_conc(win='shot1', xrange=[BAT-20, BAT+160], show=show, save=save, path=path, prefix=prefix)
        self.plot_with_conc(win='shot1_', xrange=[BAT-20, BAT+600], show=show, save=save, path=path, prefix=prefix)
        self.plot_with_conc(win='shot1__', xrange=[BAT-20, BAT+1200], show=show, save=save, path=path, prefix=prefix)
        BAT = self.aorta.p.value.BAT2
        self.plot_with_conc(win='shot2', xrange=[BAT-20, BAT+160], show=show, save=save, path=path, prefix=prefix)
        self.plot_with_conc(win='shot2_', xrange=[BAT-20, BAT+600], show=show, save=save, path=path, prefix=prefix)
        self.plot_with_conc(win='shot2__', xrange=[BAT-20, BAT+1200], show=show, save=save, path=path, prefix=prefix)

    def plot_with_conc(self, fit=True, xrange=None, win='all', show=True, save=False, path=None, legend=True, prefix=''):

        if fit is True:
            y = self.y
        else:
            y = self.yp
        if xrange is None:
            t0 = self.aorta.t[0]
            t1 = self.aorta.t[-1]
        else:
            t0 = xrange[0]
            t1 = xrange[1]
        if path is None:
            path = self.path()
        if not os.path.isdir(path):
            os.makedirs(path)
       
        ti = np.nonzero((self.aorta.t>=t0) & (self.aorta.t<=t1))[0]
        xv = np.nonzero((self.x[self.valid]>=t0) & (self.x[self.valid]<=t1))[0]
        xi = np.nonzero((self.x[self.invalid]>=t0) & (self.x[self.invalid]<=t1))[0]

        fig, (ax1, ax2) = plt.subplots(1,2,figsize=(20,8))
        fig.suptitle("model fit " + win)

        ax1.set_title('Signal')
        ax1.set(xlabel=self.xlabel, ylabel=self.ylabel)
        ax1.plot(self.x[self.valid][xv]+self.aorta.tacq/2, y[self.valid][xv], 'ro', label='valid data')
        ax1.plot(self.x[self.invalid][xi]+self.aorta.tacq/2, y[self.invalid][xi], marker='o', color='gray', label='invalid data', linestyle = 'None')
        ax1.plot(self.aorta.t[ti], self.signal_smooth()[ti], 'b-', label=self.plabel())
        if (self.R11[0] >=t0) & (self.R11[0] <=t1):
            testsignal = dcmri.signalSPGRESS(self.aorta.TR, self.p.value.FA1, self.R11[1], self.p.value.S01)
            ax1.plot(self.R11[0], testsignal, 'gx', label='test data (MOLLI)')
        if (self.R12[0] >=t0) & (self.R12[0] <=t1):
            testsignal = dcmri.signalSPGRESS(self.aorta.TR, self.p.value.FA2, self.R12[1], self.p.value.S02)            
            ax1.plot(self.R12[0], testsignal, 'gx', label='test data (MOLLI)')
        if legend:
            ax1.legend()

        ax2.set_title('Reconstructed concentration')
        ax2.set(xlabel=self.xlabel, ylabel='Concentration (mM)')
        ax2.plot(self.aorta.t[ti], 0*self.aorta.t[ti], color='black')
        ax2.plot(self.aorta.t[ti], self.cl[ti], 'b-', label='liver')
        ax2.plot(self.aorta.t[ti], self.ce[ti], 'r-', label='extracellular')
        ax2.plot(self.aorta.t[ti], self.ch[ti], 'g-', label='hepatocyte')
        if legend:
            ax2.legend()
        if save:   
            if not os.path.exists(path):
                os.makedirs(path)       
            plt.savefig(fname=os.path.join(path, prefix+'_' + win + '.png'))
        if show:
            plt.show()
        else:
            plt.close()


class OneShotTwoScan(TwoShotTwoScan):

    def estimate_p(self):

        BAT1 = self.aorta.p.value.BAT1
        Sref = dcmri.signalSPGRESS(self.aorta.TR, self.p.value.FA1, self.R10, 1)
        baseline = np.nonzero(self.x1 <= BAT1)[0]
        n0 = baseline.size
        if n0 == 0: 
            n0 = 1
        S01 = np.mean(self.y1[:n0]) / Sref
        self.p.value.S01 = S01
        self.p.value.BAT1 = BAT1

        Sref = dcmri.signalSPGRESS(self.aorta.TR, self.p.value.FA2, self.R12[1], 1)
        n0 = math.floor(60/self.aorta.tacq) # 1 minute baseline
        S02 = np.mean(self.y2[:n0]) / Sref
        self.p.value.S02 = S02

    def plot_fit(self, show=True, save=False, path=None, prefix=''):

        self.plot_with_conc(show=show, save=save, path=path, prefix=prefix)
        BAT = self.aorta.p.value.BAT1
        self.plot_with_conc(win='shot1', xrange=[BAT-20, BAT+160], show=show, save=save, path=path, prefix=prefix)
        self.plot_with_conc(win='shot1_', xrange=[BAT-20, BAT+600], show=show, save=save, path=path, prefix=prefix)
        self.plot_with_conc(win='shot1__', xrange=[BAT-20, BAT+1200], show=show, save=save, path=path, prefix=prefix)
        t0, t1 = self.aorta.x2[0], self.aorta.x2[-1]
        self.plot_with_conc(win='scan2', xrange=[t0, t1], show=show, save=save, path=path, prefix=prefix)


class TwoShotOneScan(OneShotOneScan):

    def function(self, x, p):

        R1 = self.R1()

        # Calculate signal and sample
        self.signal = dcmri.signalSPGRESS(self.aorta.TR, p.FA, R1, p.S0)
        return dcmri.sample(self.aorta.t, self.signal, x, self.aorta.tacq)

    def parameters(self):

        BAT = self.aorta.p.value.BAT1

        return [
            ['BAT1', "Bolus arrival time", BAT, "sec", 0, np.inf, False, 3], 
            ['S0', "Liver signal amplitude S0", 1000.0, "a.u.", 0, np.inf, False, 3], 
            ['FA', "Flip angle", self.aorta.FA, "deg", 0, 180, False, 4], 
            ['Ktrans', "Hepatocellular uptake rate", self.Ktrans, 'mL/sec/mL', 0, np.inf, True, 5],
            ['Th', "Hepatocellular mean transit time", 20*60., 'sec', 0, np.inf, True, 5],
            ['FpTe', "Apparent liver extracellular volume", self.veL, 'mL/mL', 0, 1, True, 3],

            ['tdel', "Gut delay time", 7.0, 'sec', 0, np.inf, False, 3],
            ['tdisp', "Gut dispersion time", 28.0, 'sec', 0, np.inf, False, 3],
        ]

    def set_R12(self, t, R1):
        self.R12 = [t, R1]

    def R1(self):

        p = self.p.value
        cv = dcmri.propagate_dd(self.aorta.t, self.aorta.cb, p.tdel, p.tdisp)
        self.cb = cv

        #cp = expconv(Te, t, cp)            # Dispersion in liver extracellular space

        cp = self.cb/self.aorta.Hct
        ne, nh = dcmri.residue_high_flow_2cfm(self.aorta.t, cp, p.Ktrans, p.Th, p.FpTe)
        self.ce = (1-p.Ktrans/self.Fp)*cp
        self.ch = nh/self.vh
        self.cl = ne + nh
        
        R1 = self.R10lit + self.aorta.rp*ne + self.rh*nh
        return R1

    def estimate_p(self):

        n0 = math.floor(60/self.aorta.tacq)
        Sref = dcmri.signalSPGRESS(self.aorta.TR, self.p.value.FA, self.R12[1], 1)
        S0 = np.mean(self.y[:n0]) / Sref
        self.p.at['S0','value'] = S0

    def plot_fit(self, show=True, save=False, path=None, prefix=''):

        self.plot_with_conc(show=show, save=save, path=path, prefix=prefix)
        BAT2 = self.aorta.p.value.BAT2
        self.plot_with_conc(win='shot1', xrange=[BAT2-20, BAT2+160], show=show, save=save, path=path, prefix=prefix)
        self.plot_with_conc(win='shot1_', xrange=[BAT2-20, BAT2+600], show=show, save=save, path=path, prefix=prefix)
        self.plot_with_conc(win='shot1__', xrange=[BAT2-20, BAT2+1200], show=show, save=save, path=path, prefix=prefix)

    def plot_with_conc(self, fit=True, xrange=None, win='all', show=True, save=False, path=None, legend=True, prefix=''):

        if fit is True:
            y = self.y
        else:
            y = self.yp
        if xrange is None:
            t0 = self.aorta.t[0]
            t1 = self.aorta.t[-1]
        else:
            t0 = xrange[0]
            t1 = xrange[1]
        if path is None:
            path = self.path()
        if path is None: 
            path = self.path()
        if not os.path.isdir(path):
            os.makedirs(path)
        ti = np.nonzero((self.aorta.t>=t0) & (self.aorta.t<=t1))[0]
        xi = np.nonzero((self.x>=t0) & (self.x<=t1))[0]

        fig, (ax1, ax2) = plt.subplots(1,2,figsize=(20,8))
        fig.suptitle("model fit " + win)

        ax1.set_title('Signal')
        ax1.set(xlabel=self.xlabel, ylabel=self.ylabel)
        ax1.plot(self.x[xi]+self.aorta.tacq/2, y[xi], 'ro', label='data')
        ax1.plot(self.aorta.t[ti], self.signal_smooth()[ti], 'b-', label=self.plabel())
        if (self.R12[0] >=t0) & (self.R12[0] <=t1):
            testsignal = dcmri.signalSPGRESS(self.aorta.TR, self.p.value.FA, self.R12[1], self.p.value.S0)
            ax1.plot(self.R12[0], testsignal, 'gx', label='test data (MOLLI)')
        if legend:
            ax1.legend()

        ax2.set_title('Reconstructed concentration')
        ax2.set(xlabel=self.xlabel, ylabel='Concentration (mM)')
        ax2.plot(self.aorta.t[ti], 0*self.aorta.t[ti], color='black')
        ax2.plot(self.aorta.t[ti], self.cl[ti], 'b-', label='liver')
        ax2.plot(self.aorta.t[ti], self.ce[ti], 'r-', label='extracellular')
        ax2.plot(self.aorta.t[ti], self.ch[ti], 'g-', label='hepatocyte')
        if legend:
            ax2.legend()
        if save:   
            if not os.path.exists(path):
                os.makedirs(path)       
            plt.savefig(fname=os.path.join(path, prefix+'_' + win + '.png'))
        if show:
            plt.show()
        else:
            plt.close()


