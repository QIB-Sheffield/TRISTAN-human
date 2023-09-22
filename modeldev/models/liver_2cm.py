import os
import math
import numpy as np
import matplotlib.pyplot as plt

import dcmri
from curvefit import CurveFit



class TwoShotTwoScan(CurveFit):

    def __init__(self, aorta):
        self.aorta = aorta
        self.aorta.signal_smooth()
        super().__init__()

    # Constants 
    vh = 0.722
    fp_half_width = 20 # sec, half width of the first pass in the aorta
    baseline2 = 60 # sec, second baseline

    @property
    def rh(self):
        field = math.floor(self.aorta.field_strength)
        if field == 1.5: return 14.6    # relaxivity of hepatocytes in Hz/mM
        if field == 3.0: return 9.8     # relaxivity of hepatocytes in Hz/mM
        if field == 4.0: return 7.6     # relaxivity of hepatocytes in Hz/mM
        if field == 7.0: return 6.0     # relaxivity of hepatocytes in Hz/mM
        if field == 9.0: return 6.1     # relaxivity of hepatocytes in Hz/mM

    @property
    def R10lit(self):
        field = math.floor(self.aorta.field_strength)
        if field == 1.5: return 1000.0/602.0     # liver R1 in 1/sec (Waterton 2021)
        if field == 3.0: return 1000.0/752.0     # liver R1 in 1/sec (Waterton 2021)
        if field == 4.0: return 1.281     # liver R1 in 1/sec (Changed from 1.285 on 06/08/2020)
        if field == 7.0: return 1.109     # liver R1 in 1/sec (Changed from 0.8350 on 06/08/2020)
        if field == 9.0: return 0.920     # per sec - liver R1 (https://doi.org/10.1007/s10334-021-00928-x)
        
    def function(self, x, p): # remove x, p args
        self.signal = self.signal_smooth()
        return dcmri.sample(self.aorta.t, self.signal, self.x, self.aorta.tacq)
    
    def signal_smooth(self):
        R1 = self.R1()
        k2 = np.nonzero(self.aorta.t >= self.aorta.x2[0])[0]
        signal = dcmri.signalSPGRESS(self.aorta.TR, self.aorta.FA, R1, self.p.value.S01)
        signal[k2] = dcmri.signalSPGRESS(self.aorta.TR, self.aorta.FA, R1[k2], self.p.value.S02)
        return signal
    
    def R1(self): # slower but more diagnostics
        p = self.p.value
        # Propagate through the gut
        cv = dcmri.propagate_dd(self.aorta.t, self.aorta.cb, p.Tg_del, p.Tg_disp)
        # Propagate through the arterial tree
        # ca = dcmri.propagate_dd(self.aorta.t, self.aorta.cb, p.Ta_del, p.Ta_disp)
        # Mix at the inlet
        # cin = p.AFF*ca + (1-p.AFF)*cv
        cin = cv
        # Propagate through the extracellular space
        self.Ce = dcmri.res_comp(self.aorta.t, p.fb*cin, np.divide(1,p.Kve+p.Khe))
        # Propagate through the hepatocytes
        self.Ch = dcmri.res_comp(self.aorta.t, p.Khe*self.Ce, np.divide(1,p.Kbh))
        # Return R1
        return p.R10 + self.aorta.rp*self.Ce + self.rh*self.Ch

    def parameters(self):
        return [
            # Signal parameters
            ['R10', "Precontrast liver R1", self.R10lit, "1/sec", 0, np.inf, False, 6],
            ['S01', "Signal amplitude S0", 1200, "a.u.", 0, np.inf, False, 4],
            ['S02', "Signal amplitude S0", 1200, "a.u.", 0, np.inf, True, 4],
            # Inlets
            ['AFF', "Arterial flow fraction", 0.0, '', 0, 1, False, 5],
            ['Ta_del', "Arterial delay time", 2.0, 'sec', 0, np.inf, False, 3],
            ['Ta_disp', "Arterial dispersion time", 4.0, 'sec', 0, np.inf, False, 3],
            ['Tg_del', "Gut delay time", 5.0, 'sec', 0, np.inf, True, 3],  
            ['Tg_disp', "Gut dispersion time", 20.0, 'sec', 0, np.inf, True, 3],
            # Liver tissue
            ['fb', "Liver blood flow", 130.0/6000, 'mL/sec/mL', 0, np.inf, True, 3],
            ['Kve', "Venous tissue uptake rate", 0.96/20, 'mL/sec/mL', 0, np.inf, True, 6],
            ['Khe', "Hepatocellular tissue uptake rate", 0.04/20, 'mL/sec/mL', 0, np.inf, True, 6],
            ['Kbh', "Biliary tissue excretion rate", 0.004/20, 'mL/sec/mL', 0, np.inf, True, 6],
        ]

    def set_export_pars(self):
        p = self.p.value
        self.export_pars = self.p.drop(['initial value','lower bound','upper bound','fit','digits'], axis=1)
        # Convert to conventional units
        self.export_pars.loc['AFF', ['value', 'unit']] = [100*p.AFF, '%']
        self.export_pars.loc['fb', ['value', 'unit']] = [6000*p.fb, 'mL/min/100mL']
        self.export_pars.loc['Kve', ['value', 'unit']] = [6000*p.Kve, 'mL/min/100mL'] # Kve Ce = Kve ve ce = fb ce
        self.export_pars.loc['Khe', ['value', 'unit']] = [6000*p.Khe, 'mL/min/100mL']
        self.export_pars.loc['Kbh', ['value', 'unit']] = [6000*p.Kbh, 'mL/min/100mL']
        # Add derived parameters 
        self.export_pars.loc['Te'] = ["Extracellular transit time", np.divide(1, p.Kve+p.Khe)/60, 'min']
        self.export_pars.loc['Th'] = ["Hepatocellular transit time", np.divide(1, p.Kbh)/60, 'min']
        self.export_pars.loc['E'] = ["Gadoxetate extraction fraction", 100*np.divide(p.Khe, p.Kve+p.Khe), '%']
        self.export_pars.loc['ve'] = ["Extracellular volume fraction", 100*np.divide(p.fb, p.Kve), 'mL/100mL']
        self.export_pars.loc['k_he'] = ["Hepatocellular uptake rate", 6000*p.Khe*np.divide(p.fb, p.Kve), 'mL/min/100mL'] # Khe Ce = Khe ve ce = khe ce
        self.export_pars.loc['k_bh'] = ["Biliary excretion rate", 6000*p.Kbh*self.vh, 'mL/min/100mL']

    def estimate_p(self):

        BAT1 = self.aorta.p.value.BAT1
        Sref = dcmri.signalSPGRESS(self.aorta.TR, self.aorta.FA, self.vR1[0], 1)
        baseline = np.nonzero(self.x1 <= BAT1-self.fp_half_width)[0]
        n0 = baseline.size
        if n0 == 0: 
            n0 = 1
        S01 = np.mean(self.y1[:n0]) / Sref
        self.p.at['S01','value'] = S01
        self.p.at['R10', 'value'] = self.vR1[0]

        Sref = dcmri.signalSPGRESS(self.aorta.TR, self.aorta.FA, self.vR1[2], 1)
        n0 = math.floor(self.baseline2/self.aorta.tacq)
        S02 = np.mean(self.y2[:n0]) / Sref
        self.p.at['S02','value'] = S02

    def set_R1(self, t, R1):
        self.tR1 = t
        self.vR1 = R1

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

    # More diagnostics
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
       
        xf, yf = self.xy_fitted()
        xi, yi = self.xy_ignored() 

        fig, (ax1, ax2) = plt.subplots(1,2,figsize=(20,8))
        fig.suptitle("model fit " + win)

        # Left plot
        ax1.set_title('Signal')
        ax1.set(xlabel=self.xlabel, ylabel=self.ylabel, xlim=[t0,t1])
        ax1.plot(xi+self.aorta.tacq/2, yi, marker='o', color='gray', label='ignored data', linestyle = 'None')
        ax1.plot(xf+self.aorta.tacq/2, yf, 'ro', label='fitted data')
        ax1.plot(self.aorta.t, self.signal_smooth(), 'b-', label='fit' )
        m1 = dcmri.signalSPGRESS(self.aorta.TR, self.aorta.FA, self.vR1[1], self.p.value.S01)
        m2 = dcmri.signalSPGRESS(self.aorta.TR, self.aorta.FA, self.vR1[2], self.p.value.S02)
        ax1.plot(self.tR1[1:], [m1,m2], 'rx', label='test data (MOLLI)')
        if legend:
            ax1.legend()

        # Right plot
        ax2.set_title('Reconstructed concentration')
        ax2.set(xlabel=self.xlabel, ylabel='Tissue concentration (mM)', xlim=[t0,t1])
        ax2.plot(self.aorta.t, 0*self.aorta.t, color='gray')
        ax2.plot(self.aorta.t, self.Ce, 'g-', label='Extracellular')
        ax2.plot(self.aorta.t, self.Ch, 'g--', label='Hepatocyte')
        ax2.plot(self.aorta.t, self.Ce+self.Ch, 'b-', label='Tissue')
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


