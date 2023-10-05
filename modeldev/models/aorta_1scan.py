import os
import math
import numpy as np
import matplotlib.pyplot as plt

import dcmri
from curvefit import CurveFit




class OneShotOneScan(CurveFit):

    xname = 'Time'
    xunit = 'sec'
    yname = 'MRI Signal'
    yunit = 'a.u.'

    def R1(self):

        p = self.p.value
        Ji = dcmri.injection(self.t, 
            self.weight, self.conc, self.dose, self.rate, p.BAT)
        _, Jb = dcmri.propagate_simple_body(self.t, Ji, 
            p.MTThl, p.El, p.MTTe, p.MTTo, p.TTDo, p.Ee, tol=self.dose_tolerance)
        self.cb = Jb*1000/p.CO  # (mM)
        return self.R10 + self.rp * self.cb

    def function(self, x, p):

        R1 = self.R1()
        self.signal = dcmri.signalSPGRESS(self.TR, p.FA, R1, p.S0)
        return dcmri.sample(self.t, self.signal, x, self.tacq)

    def parameters(self):

        return [ 
            ['FA', "Flip angle", self.FA, "deg", 0, 180, False, 4],
            ['S0', "Signal amplitude S0", 1200, "a.u.", 0, np.inf, False, 4],
            ['BAT', "Bolus arrival time", 60, "sec", 0, np.inf, True, 3],
            ['CO', "Cardiac output", 100.0, "mL/sec", 0, np.inf, True, 3], # 6 L/min = 100 mL/sec
            ['MTThl', "Heart & lung mean transit time", 8.0, "sec", 0, 30, True, 3],
            ['MTTo', "Other organs mean transit time", 20.0, "sec", 0, 60, True, 3],
            ['TTDo', "Other organs transit time dispersion", 10.0, "sec", 0, np.inf, True, 2],
            ['MTTe', "Storage compartment mean transit time", 120.0, "sec", 0, 800.0, True, 3],
            ['El', "Storage compartment leakage fraction", 0.15, "", 0, 0.50, True, 4],
            ['Ee',"Kidney & Liver Extraction fraction", 0.05,"", 0.01, 0.15, True, 4],
        ]
    
    def set_export_pars(self):
        self.export_pars = self.p.drop(['initial value','lower bound','upper bound','fit','digits'], axis=1)
        self.export_pars.drop(['FA'],inplace=True)


    def set_const(self):
        # Default values for experimental parameters
        self.t0 = 0      # Start of acquisition (sec since midnight)
        self.tacq = 1.64             # Time to acquire a single datapoint (sec)
        self.field_strength = 3.0    # Field strength (T)
        self.weight = 70.0           # Patient weight in kg
        self.conc = 0.25             # mmol/mL (https://www.bayer.com/sites/default/files/2020-11/primovist-pm-en.pdf)
        self.dose = 0.025            # mL per kg bodyweight (quarter dose)
        self.rate = 1                # Injection rate (mL/sec)
        self.TR = 3.71/1000.0        # Repetition time (sec)
        self.FA = 15.0               # Nominal flip angle (degrees)
        # Internal time resolution & acquisition time
        self.dt = 0.5                # sec
        self.tmax = 40*60.0          # Total acquisition time (sec)
        # Physiological parameters
        self.Hct = 0.45
        # Optimization parameters
        self.dose_tolerance = 0.001
        self.set_dose(0.025)

    @property
    def rp(self):
        field = math.floor(self.field_strength)
        if field == 1.5: return 8.1     # relaxivity of blood in Hz/mM
        if field == 3.0: return 6.4     # relaxivity of blood in Hz/mM
        if field == 4.0: return 6.4     # relaxivity of blood in Hz/mM
        if field == 7.0: return 6.2     # relaxivity of blood in Hz/mM
        if field == 9.0: return 6.1     # relaxivity of blood in Hz/mM 

    @property
    def t(self): # internal time
        return np.arange(0, self.tmax+self.dt, self.dt) 

    @property
    def R10lit(self):
        field = math.floor(self.field_strength)
        if field == 1.5: return 1000.0 / 1480.0         # aorta R1 in 1/sec 
        if field == 3.0: return 0.52 * self.Hct + 0.38  # Lu MRM 2004 

    def signal_smooth(self):
        R1 = self.R1()
        return dcmri.signalSPGRESS(self.TR, self.p.value.FA, R1, self.p.value.S0)
       
    def set_x(self, x, **kwargs):
        self.tacq = x[1]-x[0]
        self.tmax = x[-1] + self.tacq
        super().set_x(x, **kwargs)
        
    def set_R10(self, t, R1):
        self.R10 = R1

    def set_R11(self, t, R1):
        self.R11 = [t, R1]
        self.tmax = t + self.tacq

    def set_dose(self, dose):
        self.dose = dose
        # adjust internal time resolution
        duration = self.weight*self.dose/self.rate
        if duration == 0:
            return
        if self.dt > duration/5:
            self.dt = duration/5

    def estimate_p(self):

        BAT = self.x[np.argmax(self.y)]
        baseline = np.nonzero(self.x <= BAT-20)[0]
        n0 = baseline.size
        if n0 == 0: 
            n0 = 1
        Sref = dcmri.signalSPGRESS(self.TR, self.p.value.FA, self.R10, 1)
        S0 = np.mean(self.y[:n0]) / Sref
        self.p.at['S0','value'] = S0
        self.p.at['BAT','value'] = BAT


    def plot_fit(self, show=True, save=False, path=None, prefix=''):

        self.plot_with_conc(show=show, save=save, path=path, prefix=prefix)
        BAT = self.p.value.BAT
        self.plot_with_conc(win='pass1', xrange=[BAT-20, BAT+160], show=show, save=save, path=path, prefix=prefix)


    def plot_with_conc(self, fit=True, xrange=None, legend=True, win='all', show=True, save=False, path=None, prefix=''):

        if fit is True:
            y = self.y
        else:
            y = self.yp
        if xrange is None:
            t0 = self.t[0]
            t1 = self.R11[0]
        else:
            t0 = xrange[0]
            t1 = xrange[1]
        if path is None:
             path = self.path()
        if not os.path.isdir(path):
            os.makedirs(path)
        xf, yf = self.xy_fitted()
        xi, yi = self.xy_ignored()
        fig, (ax1, ax2) = plt.subplots(1,2,figsize=(20,8))
        fig.suptitle("model fit " + win)
        ax1.set_title('Signal')
        ax1.set(xlabel=self.xlabel, ylabel=self.ylabel, xlim=[t0,t1])
        ax1.plot(xi+self.tacq/2, yi, marker='o', color='gray', label='ignored data', linestyle = 'None')
        ax1.plot(xf+self.tacq/2, yf, 'ro', label='fitted data')
        ax1.plot(self.t, self.signal_smooth(), 'b-', label='fit' )
        m1 = dcmri.signalSPGRESS(self.TR, self.p.value.FA, self.R11[1], self.p.value.S0)
        ax1.plot([self.R11[0]], [m1], 'gx', label='test data (MOLLI)')
        if legend:
            ax1.legend()

        ax2.set_title('Reconstructed concentration')
        ax2.set(xlabel=self.xlabel, ylabel='Concentration (mM)', xlim=[t0,t1])
        ax2.plot(self.t, 0*self.t, color='black')
        ax2.plot(self.t, self.cb, 'b-', label=self.plabel())
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



