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
    vtot = 0.95
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
        k2 = np.nonzero(self.aorta.t >= self.tR1[2])[0]
        signal = dcmri.signalSPGRESS(self.aorta.TR, self.p.value.FA1_corr*self.aorta.FA, R1, self.p.value.S01)
        signal[k2] = dcmri.signalSPGRESS(self.aorta.TR, self.p.value.FA2_corr*self.aorta.FA, R1[k2], self.p.value.S02)
        return signal
    
    def R1(self): # slower but more diagnostics
        p = self.p.value
        # Propagate through the gut
        ca = dcmri.propagate_dd(self.aorta.t, self.aorta.cb, p.Tdel, p.Te)
        # Tissue concentration in the extracellular space
        self.Ce = p.ve*ca
        # Tissue concentration in the hepatocytes
        k_he = dcmri.quad(self.aorta.t, [p.k_he_i, p.k_he_m, p.k_he_f])
        Kbh = dcmri.lin(self.aorta.t, [p.Kbh_i, p.Kbh_f])
        self.Ch = dcmri.res_nscomp(self.aorta.t, k_he*ca, Kbh)
        # Return R1
        return p.R10 + self.aorta.rp*self.Ce + self.rh*self.Ch

    def parameters(self):
        return [
            # Signal parameters
            ['R10', "Precontrast liver R1", self.R10lit, "1/sec", 0, np.inf, False, 6],
            ['FA1_corr', "FA correction factor", 1, "", 0, np.inf, True, 6],
            ['FA2_corr', "FA correction factor", 1, "", 0, np.inf, True, 6],
            ['S01', "Signal amplitude S0", 1200, "a.u.", 0, np.inf, False, 6],
            ['S02', "Signal amplitude S0", 1200, "a.u.", 0, np.inf, False, 6],
            # Inlets
            ['Tdel', "Gut delay time", 5.0, 'sec', 0, 20.0, True, 6],  
            ['Te', "Extracellular transit time", 30.0, 'sec', 0, 60, True, 6],
            # Liver tissue
            ['ve', "Extracellular volume fraction", 0.2, 'mL/mL', 0.0, 1.0, True, 6],
            ['k_he_i', "Hepatocellular uptake rate", 0.2*0.04/20, 'mL/sec/mL', 0, np.inf, True, 6],
            ['k_he_m', "Hepatocellular uptake rate", 0.2*0.04/20, 'mL/sec/mL', 0, np.inf, True, 6],
            ['k_he_f', "Hepatocellular uptake rate", 0.2*0.04/20, 'mL/sec/mL', 0, np.inf, True, 6],
            ['Kbh_i', "Biliary tissue excretion rate", 1/(30*60), 'mL/sec/mL', 0, 1/(10*60), True, 6],
            ['Kbh_f', "Biliary tissue excretion rate", 1/(30*60), 'mL/sec/mL', 0, 1/(10*60), True, 6],
        ]

    def set_export_pars(self):
        p = self.p.value
        self.export_pars = self.p.drop(['initial value','lower bound','upper bound','fit','digits'], axis=1)
        # Convert to conventional units
        self.export_pars.loc['Te', ['value', 'unit']] = [p.Te/60, 'min']
        self.export_pars.loc['ve', ['value', 'unit']] = [100*p.ve, 'mL/100mL']
        self.export_pars.loc['k_he_i', ['value', 'unit']] = [6000*p.k_he_i, 'mL/min/100mL']
        self.export_pars.loc['k_he_m', ['value', 'unit']] = [6000*p.k_he_i, 'mL/min/100mL']
        self.export_pars.loc['k_he_f', ['value', 'unit']] = [6000*p.k_he_f, 'mL/min/100mL']
        self.export_pars.loc['Kbh_i', ['value', 'unit']] = [6000*p.Kbh_i, 'mL/min/100mL']
        self.export_pars.loc['Kbh_f', ['value', 'unit']] = [6000*p.Kbh_f, 'mL/min/100mL']
        # Add derived parameters 
        k_he = dcmri.quad(self.aorta.t, [p.k_he_i, p.k_he_m, p.k_he_f])
        k_he_avr = np.mean(k_he)
        k_he_var = (np.amax(k_he)-np.amin(k_he))/k_he_avr
        Kbh = dcmri.lin(self.aorta.t, [p.Kbh_i, p.Kbh_f])
        Kbh_avr = np.mean(Kbh)
        Kbh_var = (np.amax(Kbh)-np.amin(Kbh))/Kbh_avr
        self.export_pars.loc['k_he'] = ["Hepatocellular uptake rate", 6000*k_he_avr, 'mL/min/100mL']
        self.export_pars.loc['k_he_var'] = ["Hepatocellular uptake rate variance", 100*k_he_var, '%']
        self.export_pars.loc['Kbh'] = ["Biliary tissue excretion rate", 6000*Kbh_avr, 'mL/min/100mL']
        self.export_pars.loc['Kbh_var'] = ["Biliary tissue excretion rate variance", 100*Kbh_var, '%']
        self.export_pars.loc['Khe'] = ["Hepatocellular tissue uptake rate", 6000*k_he_avr/p.ve, 'mL/min/100mL']         
        self.export_pars.loc['k_bh'] = ["Biliary excretion rate", 6000*Kbh_avr*(1-p.ve), 'mL/min/100mL']
        self.export_pars.loc['Th'] = ["Hepatocellular transit time", np.divide(1, Kbh_avr)/60, 'min']
        
    def estimate_p(self):

        BAT1 = self.aorta.p.value.BAT1
        Sref = dcmri.signalSPGRESS(self.aorta.TR, self.p.value.FA1_corr*self.aorta.FA, self.vR1[0], 1)
        baseline = np.nonzero(self.x1 <= BAT1-self.fp_half_width)[0]
        n0 = baseline.size
        if n0 == 0: 
            n0 = 1
        S01 = np.mean(self.y1[:n0]) / Sref
        self.p.at['S01','value'] = S01
        self.p.at['R10', 'value'] = self.vR1[0]

        Sref = dcmri.signalSPGRESS(self.aorta.TR, self.p.value.FA2_corr*self.aorta.FA, self.vR1[2], 1)
        n0 = math.floor(self.baseline2/self.aorta.tacq)
        S02 = np.mean(self.y2[:n0]) / Sref
        self.p.at['S02','value'] = S02

    def set_R1(self, t, R1):
        self.tR1 = t
        self.vR1 = R1
        self.tR1[0] = 0

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
        m0 = dcmri.signalSPGRESS(self.aorta.TR, self.p.value.FA1_corr*self.aorta.FA, self.vR1[0], self.p.value.S01)
        m1 = dcmri.signalSPGRESS(self.aorta.TR, self.p.value.FA1_corr*self.aorta.FA, self.vR1[1], self.p.value.S01)
        m2 = dcmri.signalSPGRESS(self.aorta.TR, self.p.value.FA2_corr*self.aorta.FA, self.vR1[2], self.p.value.S02)
        ax1.plot(self.tR1, [m0,m1,m2], 'gx', label='test data (MOLLI)')
        if legend:
            ax1.legend()

        # Right plot
        ax2.set_title('Reconstructed concentration')
        ax2.set(xlabel=self.xlabel, ylabel='Tissue concentration (mM)', xlim=[t0,t1])
        ax2.plot(self.aorta.t, 0*self.aorta.t, color='gray')
        ax2.plot(self.aorta.t, self.Ce, 'g-', label='Extracellular')
        ax2.plot(self.aorta.t, self.Ch, 'g--', label='Hepatocyte')
        ax2.plot(self.aorta.t, self.Ce+self.Ch, 'b-', label=self.plabel())
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


