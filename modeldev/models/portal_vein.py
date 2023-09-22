import os
import math
import numpy as np
import matplotlib.pyplot as plt

import dcmri
from curvefit import CurveFit


class OneShotOneScan(CurveFit):

    def function(self, x, p):

        R1 = self.R1()
        self.signal = dcmri.signalSPGRESS(self.aorta.TR, p.FA, R1, p.S0)
        return dcmri.sample(self.aorta.t, self.signal, x, self.aorta.tacq)

    def parameters(self):

        return [
            ['S0', "Signal amplitude S0", 1200, "a.u.", 0, np.inf, False, 4],
            ['FA', "Flip angle", self.aorta.FA, "deg", 0, 180, False, 4],
            ['tdel', "Gut delay time", 7.0, 'sec', 0, np.inf, True, 3],
            ['tdisp', "Gut dispersion time", 28.0, 'sec', 0, np.inf, True, 3],
            ['Vport', "Portal venous volume fraction", 1.0, '', 0, 1, True, 3],
        ]

    def set_export_pars(self):
        self.export_pars = self.p.drop(['initial value','lower bound','upper bound','fit','digits'], axis=1)
        self.export_pars.drop(['FA'],inplace=True)

    def __init__(self, aorta):
        self.aorta = aorta
        self.aorta.signal_smooth()
        super().__init__()

    def R1(self):
        p = self.p.value
        self.cv = dcmri.propagate_dd(self.aorta.t, self.aorta.cb, p.tdel, p.tdisp)
        return self.R10 + self.aorta.rp*p.Vport*self.cv

    def signal_smooth(self):
        R1 = self.R1()
        return dcmri.signalSPGRESS(self.aorta.TR, self.p.value.FA, R1, self.p.value.S0)

    def set_R10(self, t, R1):
        self.R10 = R1

    def set_R11(self, t, R1):
        self.R11 = [t, R1]

    def estimate_p(self):
        baseline = np.nonzero(self.x <= self.aorta.p.value.BAT-5)[0] # Assumes baseline is until 5sec before BAT
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

        ax2.set_title('Blood concentration')
        ax2.set(xlabel=self.xlabel, ylabel='Concentration (mM)')
        ax2.plot(self.aorta.t[ti], 0*self.aorta.t[ti], color='black')
        ax2.plot(self.aorta.t[ti], self.aorta.cb[ti], 'r-', label='aorta')
        ax2.plot(self.aorta.t[ti], self.cv[ti], 'b-', label='portal vein')
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

    def signal_smooth(self):

        R1 = self.R1()
        k2 = np.nonzero(self.aorta.t >= self.aorta.x2[0])[0]
        signal = dcmri.signalSPGRESS(self.aorta.TR, self.p.value.FA1, R1, self.p.value.S01)
        signal[k2] = dcmri.signalSPGRESS(self.aorta.TR, self.p.value.FA2, R1[k2], self.p.value.S02)
        return signal

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
            ['tdel', "Gut delay time", 7.0, 'sec', 0, np.inf, True, 3],
            ['tdisp', "Gut dispersion time", 28.0, 'sec', 0, np.inf, True, 3],
            ['Vport', "Portal venous volume fraction", 1.0, '', 0, 1, True, 3],
        ]

    def set_export_pars(self):
        self.export_pars = self.p.drop(['initial value','lower bound','upper bound','fit','digits'], axis=1)
        self.export_pars.drop(['FA1','FA2'],inplace=True)
    
    def estimate_p(self):

        BAT1 = self.aorta.p.value.BAT1
        Sref = dcmri.signalSPGRESS(self.aorta.TR, self.p.value.FA1, self.R10, 1)
        baseline = np.nonzero(self.x1 <= BAT1-20)[0]
        n0 = baseline.size
        if n0 == 0: 
            n0 = 1
        S01 = np.mean(self.y1[:n0]) / Sref
        self.p.at['S01','value'] = S01

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

        ax2.set_title('Blood concentration')
        ax2.set(xlabel=self.xlabel, ylabel='Concentration (mM)')
        ax2.plot(self.aorta.t[ti], 0*self.aorta.t[ti], color='black')
        ax2.plot(self.aorta.t[ti], self.aorta.cb[ti], 'r-', label='aorta')
        ax2.plot(self.aorta.t[ti], self.cv[ti], 'b-', label='portal vein')
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
