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
            ['S0', "Signal amplitude S0", 1200, "a.u.", 0, np.inf, False, 4],
            ['FA', "Flip angle", self.FA, "deg", 0, 180, False, 4],
            ['BAT', "Bolus arrival time", 60, "sec", 0, np.inf, True, 3],
            ['CO', "Cardiac output", 3000.0/60.0, "mL/sec", 0, np.inf, True, 3],
            ['MTThl', "Heart & lung mean transit time", 6.0, "sec", 0, np.inf, True, 2],
            ['MTTo', "Other organs mean transit time", 20.0, "sec", 0, np.inf, True, 2],
            ['TTDo', "Other organs transit time dispersion", 10.0, "sec", 0, np.inf, True, 2],
            ['MTTe', "Extravascular mean transit time", 120.0, "sec", 0, np.inf, True, 3],
            ['El', "Leakage fraction", 0.5, "", 0, 1, True, 3],
            ['Ee', "Extraction fraction", 0.2,"", 0, 1, True, 3],
        ]
    
    def set_export_pars(self):
        self.export_pars = self.p.drop(['initial value','lower bound','upper bound','fit','digits'], axis=1)
        self.export_pars.drop(['FA'],inplace=True)

    # Default values for experimental parameters
    tacq = 1.64             # Time to acquire a single datapoint (sec)
    field_strength = 3.0    # Field strength (T)
    weight = 70.0           # Patient weight in kg
    conc = 0.25             # mmol/mL (https://www.bayer.com/sites/default/files/2020-11/primovist-pm-en.pdf)
    dose = 0.025            # mL per kg bodyweight (quarter dose)
    rate = 1                # Injection rate (mL/sec)
    TR = 3.71/1000.0        # Repetition time (sec)
    FA = 15.0               # Nominal flip angle (degrees)

    # Internal time resolution & acquisition time
    dt = 0.5                # sec
    tmax = 40*60.0          # Total acquisition time (sec)

    # Physiological parameters
    Hct = 0.45

    # Optimization parameters
    dose_tolerance = 0.001

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
        #self.x = x
        self.tacq = x[1]-x[0]
        self.tmax = x[-1] + self.tacq
        #self.set_valid()
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

    def plot_with_conc(self, fit=True, xrange=None, win='all', show=True, save=False, path=None, legend=True, prefix=''):

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
        x, y = self.xy_valid()
        ti = np.nonzero((self.t>=t0) & (self.t<=t1))[0]
        xv = np.nonzero((x>=t0) & (x<=t1))[0]
        xi = np.nonzero((self.x[self.invalid]>=t0) & (self.x[self.invalid]<=t1))[0]

        fig, (ax1, ax2) = plt.subplots(1,2,figsize=(20,8))
        fig.suptitle("model fit " + win)

        ax1.set_title('Signal')
        ax1.set(xlabel=self.xlabel, ylabel=self.ylabel)
        ax1.plot(x[xv]+self.tacq/2, y[xv], 'ro', label='data (fitted)')
        ax1.plot(self.x[self.invalid][xi]+self.tacq/2, self.y[self.invalid][xi], marker='o', color='gray', label='data (ignored)', linestyle = 'None')
        ax1.plot(self.t[ti], self.signal_smooth()[ti], 'b-', label='fit')
        if (self.R11[0] >=t0) & (self.R11[0] <=t1):
            testsignal = dcmri.signalSPGRESS(self.TR, self.p.value.FA, self.R11[1], self.p.value.S0)
            ax1.plot(self.R11[0], testsignal, 'gx', label='test data (MOLLI)')
        if legend:
            ax1.legend()

        ax2.set_title('Reconstructed concentration')
        ax2.set(xlabel=self.xlabel, ylabel='Concentration (mM)')
        ax2.plot(self.t[ti], 0*self.t[ti], color='black')
        ax2.plot(self.t[ti], self.cb[ti], 'b-', label=self.plabel())
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


class TwoShotTwoScan(OneShotOneScan):

    def function(self, x, p):

        R1 = self.R1()
        k2 = np.nonzero(self.t >= self.x2[0])[0]

        # Calculate signal
        self.signal = dcmri.signalSPGRESS(self.TR, p.FA1, R1, p.S01)
        self.signal[k2] = dcmri.signalSPGRESS(self.TR, p.FA2, R1[k2], p.S02)
        return dcmri.sample(self.t, self.signal, x, self.tacq)

    def parameters(self):

        return [
            ['S01', "Signal amplitude S0", 1200, "a.u.", 0, np.inf, False, 4],
            ['FA1', "Flip angle 1", self.FA, "deg", 0, 180, False, 4],
            ['S02', "Signal amplitude S0", 1200, "a.u.", 0, np.inf, True, 4],
            ['FA2', "Flip angle 2", self.FA, "deg", 0, 180, False, 4],
            ['BAT1', "Bolus arrival time", 60, "sec", 0, np.inf, True, 3],
            ['BAT2', "Bolus arrival time - second shot", 10000, "sec", 0, np.inf, True, 3],
            ['CO', "Cardiac output", 3000.0/60.0, "mL/sec", 0, np.inf, True, 3],
            ['MTThl', "Heart & lung mean transit time", 6.0, "sec", 0, np.inf, True, 2],
            ['MTTo', "Other organs mean transit time", 20.0, "sec", 0, np.inf, True, 2],
            ['TTDo', "Other organs transit time dispersion", 10.0, "sec", 0, np.inf, True, 3],
            ['MTTe', "Extravascular mean transit time", 120.0, "sec", 0, np.inf, True, 3],
            ['El', "Leakage fraction", 0.5, "", 0, 1, True, 4],
            ['Ee',"Extraction fraction", 0.2,"", 0, 1, True, 4],
        ]
    
    def set_export_pars(self):
        self.export_pars = self.p.drop(['initial value','lower bound','upper bound','fit','digits'], axis=1)
        self.export_pars.drop(['FA1', 'FA2'],inplace=True)

    def R1(self):

        # k1 = np.nonzero(self.t <= self.R11[0])[0][-1]
        # Delta_R1b = (Jb/Jb[k1])*(self.R11[1]-self.R10)
        # self.cb = Delta_R1b/self.rp
        # R1 = self.R10 + Delta_R1b

        p = self.p.value
        Ji = dcmri.injection(self.t, 
            self.weight, self.conc, self.dose[0], self.rate, p.BAT1, dose2=self.dose[1], start2=p.BAT2)
        _, Jb = dcmri.propagate_simple_body(self.t, Ji, 
            p.MTThl, p.El, p.MTTe, p.MTTo, p.TTDo, p.Ee, tol=self.dose_tolerance)
        self.cb = Jb*1000/p.CO  # (mM)
        return self.R10 + self.rp * self.cb

    def signal_smooth(self):
        
        R1 = self.R1()
        k2 = np.nonzero(self.t >= self.x2[0])[0]
        signal = dcmri.signalSPGRESS(self.TR, self.p.value.FA1, R1, self.p.value.S01)
        signal[k2] = dcmri.signalSPGRESS(self.TR, self.p.value.FA2, R1[k2], self.p.value.S02)
        return signal

    def estimate_p(self):

        BAT1 = self.x1[np.argmax(self.y1)]
        Sref = dcmri.signalSPGRESS(self.TR, self.p.value.FA1, self.R10, 1)
        baseline = np.nonzero(self.x1 <= BAT1-20)[0]
        n0 = baseline.size
        if n0 == 0: 
            n0 = 1
        S01 = np.mean(self.y1[:n0]) / Sref
        self.p.at['S01','value'] = S01
        self.p.at['BAT1','value'] = BAT1

        BAT2 = self.x2[np.argmax(self.y2)]
        Sref = dcmri.signalSPGRESS(self.TR, self.p.value.FA2, self.R11[1], 1)
        n0 = math.floor(60/self.tacq) # 1 minute baseline
        S02 = np.mean(self.y2[:n0]) / Sref
        self.p.at['S02','value'] = S02
        self.p.at['BAT2','value'] = BAT2

    def set_R11(self, t, R1):
        self.R11 = [t, R1]

    def set_R12(self, t, R1):
        self.R12 = [t, R1]

    def set_dose(self, dose1, dose2):
        self.dose = [dose1, dose2]
        # adjust internal time resolution
        duration1 = self.weight*dose1/self.rate
        duration2 = self.weight*dose2/self.rate
        min_duration = np.amin([duration1, duration2])
        if min_duration == 0:
            return
        if self.dt > min_duration/5:
            self.dt = min_duration/5

    def set_x(self, x1, x2, **kwargs):
        self.tacq = x1[1] - x1[0]
        self.tmax = x2[-1] + self.tacq
        self.x1 = x1
        self.x2 = x2
        x = np.append(x1, x2)
        super().set_x(x, **kwargs)

    def set_y(self, y1, y2):

        self.y = np.append(y1, y2)
        self.y1 = y1
        self.y2 = y2

    def plot_fit(self, xrange=None, show=True, save=False, path=None, prefix=''):

        self.plot_with_conc(show=show, save=save, path=path, prefix=prefix)
        BAT = self.p.value.BAT1
        self.plot_with_conc(win='shot1', xrange=[BAT-20, BAT+160], show=show, save=save, path=path, prefix=prefix)
        BAT = self.p.value.BAT2
        self.plot_with_conc(win='shot2', xrange=[BAT-20, BAT+160], show=show, save=save, path=path, prefix=prefix)

    def plot_with_conc(self, fit=True, xrange=None, legend=True, win='all', show=True, save=False, path=None, prefix=''):

        if fit is True:
            y = self.y
        else:
            y = self.yp
        if xrange is None:
            t0 = self.t[0]
            t1 = self.t[-1]
        else:
            t0 = xrange[0]
            t1 = xrange[1]
        if path is None:
            path = self.path()
        xf, yf = self.xy_fitted()
        xi, yi = self.xy_ignored()
        fig, (ax1, ax2) = plt.subplots(1,2,figsize=(20,8))
        fig.suptitle("model fit " + win)

        ax1.set_title('Signal')
        ax1.set(xlabel=self.xlabel, ylabel=self.ylabel, xlim=[t0,t1])
        ax1.plot(xi+self.tacq/2, yi, marker='o', color='gray', label='ignored data', linestyle = 'None')
        ax1.plot(xf+self.tacq/2, yf, 'ro', label='fitted data')
        ax1.plot(self.t, self.signal_smooth(), 'b-', label='fit' )
        m1 = dcmri.signalSPGRESS(self.TR, self.p.value.FA1, self.R11[1], self.p.value.S01)
        m2 = dcmri.signalSPGRESS(self.TR, self.p.value.FA2, self.R12[1], self.p.value.S01)
        ax1.plot([self.R11[0],self.R12[0]], [m1,m2], 'gx', label='test data (MOLLI)')
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
            plt.savefig(fname=os.path.join(path, prefix + '_' + win + '.png'))
        if show:
            plt.show()
        else:
            plt.close()


class OneShotTwoScan(TwoShotTwoScan):

    def parameters(self):

        return [
            ['S01', "Signal amplitude S0", 1200, "a.u.", 0, np.inf, False, 4],
            ['FA1', "Flip angle 1", self.FA, "deg", 0, 180, False, 4],
            ['S02', "Signal amplitude S0", 1200, "a.u.", 0, np.inf, False, 4],
            ['FA2', "Flip angle 1", self.FA, "deg", 0, 180, False, 4],
            ['BAT1', "Bolus arrival time", 60, "sec", 0, np.inf, True, 3],
            ['CO', "Cardiac output", 3000.0/60.0, "mL/sec", 0, np.inf, True, 3],
            ['MTThl', "Heart & lung mean transit time", 6.0, "sec", 0, np.inf, True, 2],
            ['MTTo', "Other organs mean transit time", 20.0, "sec", 0, np.inf, True, 2],
            ['TTDo', "Other organs transit time dispersion", 10.0, "sec", 0, np.inf, True, 3],
            ['MTTe', "Extravascular mean transit time", 120.0, "sec", 0, np.inf, True, 3],
            ['El', "Leakage fraction", 0.5, "", 0, 1, True, 4],
            ['Ee',"Extraction fraction", 0.2,"", 0, 1, True, 4],
        ]

    def R1(self):

        p = self.p.value
        Ji = dcmri.injection(self.t, 
            self.weight, self.conc, self.dose, self.rate, p.BAT1)
        _, Jb = dcmri.propagate_simple_body(self.t, Ji, 
            p.MTThl, p.El, p.MTTe, p.MTTo, p.TTDo, p.Ee, tol=self.dose_tolerance)
        self.cb = Jb*1000/p.CO  # (mM)
        return self.R10 + self.rp * self.cb
    
    def set_dose(self, dose):
        self.dose = dose
        # adjust internal time resolution
        duration = self.weight*self.dose/self.rate
        if duration == 0:
            return
        if self.dt > duration/5:
            self.dt = duration/5

    def estimate_p(self):

        BAT1 = self.x1[np.argmax(self.y1)]
        Sref = dcmri.signalSPGRESS(self.TR, self.p.value.FA1, self.R10, 1)
        baseline = np.nonzero(self.x1 <= BAT1-20)[0]
        n0 = baseline.size
        if n0 == 0: 
            n0 = 1
        S01 = np.mean(self.y1[:n0]) / Sref
        self.p.at['S01','value'] = S01
        self.p.at['BAT1','value'] = BAT1

        Sref = dcmri.signalSPGRESS(self.TR, self.p.value.FA2, self.R11[1], 1)
        n0 = math.floor(60/self.tacq) # 1 minute baseline
        S02 = np.mean(self.y2[:n0]) / Sref
        self.p.at['S02','value'] = S02

    def plot_fit(self, show=True, save=False, path=None, prefix=''):

        self.plot_with_conc(show=show, save=save, path=path, prefix=prefix)
        BAT = self.p.value.BAT1
        self.plot_with_conc(win='pass1', xrange=[BAT-20, BAT+160], show=show, save=save, path=path, prefix=prefix)


class TwoShotOneScan(OneShotOneScan):

    def function(self, x, p):

        R1 = self.R1()
        self.signal = dcmri.signalSPGRESS(self.TR, p.FA, R1, p.S0)
        return dcmri.sample(self.t, self.signal, x, self.tacq)

    def parameters(self):

        return [
            ['S0', "Signal amplitude S0", 1200, "a.u.", 0, np.inf, False, 4],
            ['FA', "Flip angle", self.FA, "deg", 0, 180, False, 4],
            ['BAT1', "Bolus arrival time", 60, "sec", 0, np.inf, False, 3],
            ['BAT2', "Bolus arrival time - second shot", 10000, "sec", 0, np.inf, True, 3],
            ['CO', "Cardiac output", 3000.0/60.0, "mL/sec", 0, np.inf, True, 3],
            ['MTThl', "Heart & lung mean transit time", 6.0, "sec", 0, np.inf, True, 2],
            ['MTTo', "Other organs mean transit time", 20.0, "sec", 0, np.inf, True, 2],
            ['TTDo', "Other organs transit time dispersion", 10.0, "sec", 0, np.inf, True, 3],
            ['MTTe', "Extravascular mean transit time", 120.0, "sec", 0, np.inf, True, 3],
            ['El', "Leakage fraction", 0.5, "", 0, 1, True, 4],
            ['Ee',"Extraction fraction", 0.2,"", 0, 1, True, 4],
        ]

    def R1(self):

        p = self.p.value
        Ji = dcmri.injection(self.t, 
            self.weight, self.conc, self.dose[0], self.rate, p.BAT1, dose2=self.dose[1], start2=p.BAT2)
        _, Jb = dcmri.propagate_simple_body(self.t, Ji, 
            p.MTThl, p.El, p.MTTe, p.MTTo, p.TTDo, p.Ee, tol=self.dose_tolerance)
        self.cb = Jb*1000/p.CO  # (mM)
        t0 = np.nonzero(self.t >= self.R12[0])[0][0]
        R10 = self.R12[1] - self.rp*self.cb[t0]
        return R10 + self.rp*self.cb

    def set_R12(self, t, R1):
        self.R12 = [t, R1]

    def set_dose(self, dose1, dose2):
        self.dose = [dose1, dose2]
        # adjust internal time resolution
        duration1 = self.weight*dose1/self.rate
        duration2 = self.weight*dose2/self.rate
        min_duration = np.amin([duration1, duration2])
        if min_duration == 0:
            return
        if self.dt > min_duration/5:
            self.dt = min_duration/5

    def estimate_p(self):

        BAT2 = self.x[np.argmax(self.y)]
        n0 = math.floor(60/self.tacq)
        Sref = dcmri.signalSPGRESS(self.TR, self.p.value.FA, self.R12[1], 1)
        S0 = np.mean(self.y[:n0]) / Sref
        self.p.at['S0','value'] = S0
        self.p.at['BAT2','value'] = BAT2

    def plot_fit(self, show=True, save=False, path=None, prefix=''):

        self.plot_with_conc(show=show, save=save, path=path, prefix=prefix)
        BAT = self.p.value.BAT2
        self.plot_with_conc(win='shot2', xrange=[BAT-20, BAT+160], show=show, save=save, path=path, prefix=prefix)

    def plot_with_conc(self, fit=True, xrange=None, win='all', show=True, save=False, path=None, legend=True, prefix=''):

        if fit is True:
            y = self.y
        else:
            y = self.yp
        if xrange is None:
            t0 = self.t[0]
            t1 = self.t[-1]
        else:
            t0 = xrange[0]
            t1 = xrange[1]
        if path is None:
            path = self.path()
        if not os.path.isdir(path):
            os.makedirs(path)
        ti = np.nonzero((self.t>=t0) & (self.t<=t1))[0]
        xi = np.nonzero((self.x>=t0) & (self.x<=t1))[0]

        fig, (ax1, ax2) = plt.subplots(1,2,figsize=(20,8))
        fig.suptitle("model fit " + win)

        ax1.set_title('Signal')
        ax1.set(xlabel=self.xlabel, ylabel=self.ylabel)
        ax1.plot(self.x[xi]+self.tacq/2, y[xi], 'ro', label='data')
        ax1.plot(self.t[ti], self.signal_smooth()[ti], 'b-', label='fit')
        if (self.R12[0] >=t0) & (self.R12[0] <=t1):
            testsignal = dcmri.signalSPGRESS(self.TR, self.p.value.FA, self.R12[1], self.p.value.S0)
            ax1.plot(self.R12[0], testsignal, 'gx', label='test data (MOLLI)')
        if legend:
            ax1.legend()

        ax2.set_title('Reconstructed concentration')
        ax2.set(xlabel=self.xlabel, ylabel='Concentration (mM)')
        ax2.plot(self.t[ti], 0*self.t[ti], color='black')
        ax2.plot(self.t[ti], self.cb[ti], 'b-', label=self.plabel())
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
