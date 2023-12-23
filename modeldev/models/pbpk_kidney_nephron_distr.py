import os
import math
import numpy as np
import matplotlib.pyplot as plt

import dcmri
import models.lib as lib

class Model(lib.FitFunc):
    def __init__(self, 
            # Constants needed to predict pseudocontinuous signal
            J_aorta = None,
            TR = 3.71/1000.0,           # Repetition time (sec)
            FA = 15.0,                  # Nominal flip angle (degrees)
            dt = 0.5,                   # Internal time resolution (sec)
            tmax = 40*60,               # Acquisition time (sec)
            field_strength = 3.0,       # Field strength (T)
            Hct = 0.45,
            # Additional constants used to predict data
            tdce = None,
            t0 = None,
            # Constants used in model fitting
            Sdce = None,
            R1molli = None,
            callback = False,
            ptol = 1e-8,
            dcevalid = None,
            tstart = None, 
            tstop = None, 
            kidney_volume = 300, #mL
            CO = 100, #mL/sec
            # Constants used in model fitting
            BAT = None,  
        ):
        self.dt = dt
        self.tmax = tmax 
        self.field_strength = field_strength
        self.Hct = Hct
        # Constants needed for data prediction
        self.tdce = tdce
        self.t0 = t0
        self.callback = callback
        # Constants needed for model fitting
        self.Sdce = Sdce
        self.R1 = R1molli
        self.ftol = 1e-8
        self.ptol = ptol
        self.gtol = 1e-8
        self._set_xind(dcevalid, tstart, tstop)
        # Essential constants
        self.J_aorta = J_aorta
        self.kidney_volume = kidney_volume
        self.CO=CO
        # Constants needed for model fitting
        self.BAT = BAT
        self.set_pars(TR,FA)
        self._set_df()

    def predict_R1(self):
        p = self.p.value
        t = self.t()
        E_k = p.F_k/(1+p.F_k)
        Kvp = 1/p.Tv
        Kp = Kvp/(1-E_k)
        J_kidneys = p.FF_k*self.J_aorta
        J_kidneys = dcmri.prop_plug(t, J_kidneys, p.Ta)
        Np = dcmri.res_nscomp(t, J_kidneys, Kp)
        Nt = dcmri.res_plug_distr(t, E_k*Kp*Np, p.Tt, p.Dt, n=50)
        self.Cp = 1000*Np/self.kidney_volume # mM
        self.Ct = 1000*Nt/self.kidney_volume # mM
        # Return R
        rp = lib.rp(self.field_strength)
        R1k = p.R10k + rp*self.Cp + rp*self.Ct
        return R1k
    
    def pars(self):
        return [
            ['R10k', "Baseline R1 (kidneys)", lib.R1_kidney(), "1/sec", 0, np.inf, False, 6],
            ['Ta',"Arterial delay time", 1.0,"sec", 0.0, 3.0, True, 6],
            ['Tv',"Vascular mean transit time", 3.0,"sec", 1.0, 7.0, True, 6],
            ['FF_k', "Kidney flow fraction", 0.2, "", 0.01, 1, True, 6],
            ['F_k',"Kidney filtration fraction", 0.10, "", 0.0, 1.0, True, 6],
            ['Tt', "Tubular mean transit time", 1.3*60, 'sec', 0, np.inf, True, 9],
            ['Dt', "Tubular transit time dispersion", 5, '%', 5, 95, True, 9],
        ]
    
    def export_pars(self, export_pars):
        p = self.p.value
        # Convert to conventional units
        export_pars.loc['FF_k', ['value', 'unit']] = [100*p.FF_k, '%']
        export_pars.loc['F_k', ['value', 'unit']] = [100*p.F_k, '%']
        # export_pars.loc['Tt0', ['value', 'unit']] = [p.Tt0/60, 'min']
        # export_pars.loc['Tt1', ['value', 'unit']] = [p.Tt1/60, 'min']
        # export_pars.loc['fn0', ['value', 'unit']] = [100*p.fn0, '%']
        # Add derived
        E_k = np.divide(p.F_k, 1+p.F_k)
        export_pars.loc['E_k'] = ["Kidney extraction fraction", 100*E_k, '%']
        export_pars.loc['GFR'] = ["Glomerular Filtration Rate", 60*p.F_k*p.FF_k*self.CO*(1-self.Hct), 'mL/min']  
        export_pars.loc['RBF'] = ["Renal blood flow", 60*p.FF_k*self.CO, 'mL/min']
        export_pars.loc['RP'] = ["Renal perfusion", 6000*p.FF_k*self.CO/self.kidney_volume, 'mL/min/100mL']
        export_pars.loc['RBV'] = ["Renal blood volume", 100*p.FF_k*self.CO*p.Tv/self.kidney_volume, 'mL/100mL']
        return export_pars

    def plot_conc_fit(self, ax2, xlim, legend):
        t = self.t()
        ax2.set_title('Reconstructed concentration')
        ax2.set(xlabel='Time (sec)', ylabel='Tissue concentration (mM)', xlim=xlim)
        ax2.plot(t, 0*t, color='gray')
        ax2.plot(t, self.Cp, 'r-', label='Plasma')
        ax2.plot(t, self.Ct, 'g-', label='Tubuli')
        ax2.plot(t, self.Cp+self.Ct, 'b-', label=self.plabel())
        if legend:
            ax2.legend()

    def _fit_function(self, _, *params):
        self.it += 1 
        if self.callback:
            p = ' '.join([str(p) for p in params])
            print('Fitting ' + self.__class__.__name__ + ', iteration: ' + str(self.it))
            print('>> Parameter values: ' + p)
        self.p.loc[self.p.fit,'value'] = params
        self.predict_data() 
        return self.yp[self.valid][self.xind]
    
    def goodness(self):
        self.predict_data()
        _, yref = self.xy_fitted() 
        ydist = np.linalg.norm(self.yp[self.valid][self.xind] - yref)
        loss = 100*ydist/np.linalg.norm(yref)
        return loss 
    
    def pars_2scan(self, TR, FA):
        return [
            ['TR', "Repetition time", TR, "sec", 0, np.inf, False, 4],
            ['FA1', "Flip angle 1", FA, "deg", 0, 180, False, 4],
            ['FA2', "Flip angle 2", FA, "deg", 0, 180, False, 4],
            ['S01k', "Signal amplitude S0", 1200, "a.u.", 0, np.inf, False, 4],
            ['S02k', "Signal amplitude S0", 1200, "a.u.", 0, np.inf, True, 4],
        ]
    
    def pars_1scan(self, TR, FA):
        return [
            ['TR', "Repetition time", TR, "sec", 0, np.inf, False, 4],
            ['FA', "Flip angle", FA, "deg", 0, 180, False, 4],
            ['S0k', "Signal amplitude S0", 1200, "a.u.", 0, np.inf, False, 4],
        ]
    
    def plot_fit_tissue_1scan(self, show=True, save=False, path=None, prefix=''):
        self.plot_with_conc(show=show, save=save, path=path, prefix=prefix)
        BAT = self.BAT
        self.plot_with_conc(win='win', xrange=[BAT-20, BAT+120], show=show, save=save, path=path, prefix=prefix)

    def plot_fit_tissue_2scan(self, show=True, save=False, path=None, prefix=''):
        self.plot_with_conc(show=show, save=save, path=path, prefix=prefix)
        BAT = self.BAT[0]
        self.plot_with_conc(win='shot1', xrange=[BAT-20, BAT+160], show=show, save=save, path=path, prefix=prefix)
        self.plot_with_conc(win='shot1_', xrange=[BAT-20, BAT+600], show=show, save=save, path=path, prefix=prefix)
        self.plot_with_conc(win='shot1__', xrange=[BAT-20, BAT+1200], show=show, save=save, path=path, prefix=prefix)
        BAT = self.BAT[1]
        self.plot_with_conc(win='shot2', xrange=[BAT-20, BAT+160], show=show, save=save, path=path, prefix=prefix)
        self.plot_with_conc(win='shot2_', xrange=[BAT-20, BAT+600], show=show, save=save, path=path, prefix=prefix)
        self.plot_with_conc(win='shot2__', xrange=[BAT-20, BAT+1200], show=show, save=save, path=path, prefix=prefix)

    def estimate_p_tissue_1scan(self):
        self.p.at['R10k','value'] = self.R1
        p = self.p.value
        Sref = dcmri.signalSPGRESS(p.TR, p.FA, p.R10k, 1)
        baseline = np.nonzero(self.tdce <= self.BAT)[0]
        n0 = baseline.size
        if n0 == 0: 
            n0 = 1
        S0 = np.mean(self.Sdce[:n0]) / Sref
        self.p.at['S0k','value'] = S0

    def estimate_p_tissue_2scan(self):
        self.p.at['R10k','value'] = self.R1[0]
        p = self.p.value

        k1 = self.tdce < self.t0[1]-self.t0[0]
        tdce1 = self.tdce[k1]
        Sdce1 = self.Sdce[k1]
        BAT = self.BAT[0]
        Sref = dcmri.signalSPGRESS(p.TR, p.FA1, p.R10k, 1)
        baseline = np.nonzero(tdce1 <= BAT)[0]
        n0 = baseline.size
        if n0 == 0: 
            n0 = 1
        S01 = np.mean(Sdce1[:n0]) / Sref
        self.p.at['S01k','value'] = S01

        k2 = self.tdce >= self.t0[1]-self.t0[0]
        tdce2 = self.tdce[k2]
        Sdce2 = self.Sdce[k2]
        Sref = dcmri.signalSPGRESS(p.TR, p.FA2, self.R1[1], 1)
        n0 = math.floor(60/(tdce2[1]-tdce2[0])) # 1 minute baseline
        S02 = np.mean(Sdce2[:n0]) / Sref
        self.p.at['S02k','value'] = S02

    def plot_with_conc(self, xrange=None, legend=True, win='all', show=True, save=False, path=None, prefix=''):
        t = self.t()
        if xrange is None:
            xrange = [t[0],t[-1]]
        fig, (ax1, ax2) = plt.subplots(1,2,figsize=(20,8))
        fig.suptitle("model fit " + win)
        self.plot_data_fit(ax1, xrange, legend)
        self.plot_conc_fit(ax2, xrange, legend)
        if save:   
            path = self.save_path(path)      
            plt.savefig(fname=os.path.join(path, prefix + '_' + win + '.png'))
        if show:
            plt.show()
        else:
            plt.close()

    def plot_data_fit(self, ax1, xlim, legend):
        xf, yf = self.xy_fitted()
        xi, yi = self.xy_ignored()
        tacq = self.tdce[1]-self.tdce[0]
        ax1.set_title('Signal')
        ax1.set(xlabel='Time (sec)', ylabel='MR Signal (a.u.)', xlim=xlim)
        ax1.plot(xi+tacq/2, yi, marker='o', color='gray', label='ignored data', linestyle = 'None')
        ax1.plot(xf+tacq/2, yf, 'go', label='fitted data')
        ax1.plot(self.t(), self.predict_signal(), 'g-', label='fit' )
        if legend:
            ax1.legend()

    def xy_fitted(self):
        x = self.tdce[self.valid][self.xind]
        y = self.Sdce[self.valid][self.xind]
        return x, y 
    
    def xy_ignored(self):
        valid = np.full(self.tdce.shape, False)
        valid[self.valid[0][self.xind]] = True
        ind = np.where(valid==False)
        return self.tdce[ind], self.Sdce[ind]

    def predict_data(self):
        signal = self.predict_signal()
        x = self.tdce
        tacq = self.tdce[1]-self.tdce[0]
        self.yp = dcmri.sample(self.t(), signal, x, tacq)
        return self.yp
    
    def get_valid(self, valid, tstart, tstop):
        if valid is None:
            valid = np.full(self.tdce.shape, True)
        if tstart is None:
            tstart = np.amin(self.tdce)
        if tstop is None:
            tstop = np.amax(self.tdce)
        t = self.tdce[np.where(valid)]
        return valid, np.nonzero((t>=tstart) & (t<=tstop))[0]

    def _set_xind(self, valid, tstart, tstop):
        if self.tdce is None:
            return
        if tstart is None:
            tstart = np.amin(self.tdce)
        if tstop is None:
            tstop = np.amax(self.tdce)
        vb, xb = self.get_valid(valid, tstart, tstop)
        self.valid = np.where(vb)
        self.xind = xb

    def t(self): # internal time
        return np.arange(0, self.tmax+self.dt, self.dt)
    


class OneScan(Model):

    def predict_signal(self):
        R1 = self.predict_R1()
        p = self.p.value
        signal = dcmri.signalSPGRESS(p.TR, p.FA, R1, p.S0k)
        return signal
    
    def set_pars(self, TR, FA):
        self.p = self.pars_1scan(TR, FA) + super().pars()

    def export_pars(self, export_pars):
        export_pars.drop(['TR','FA','S0k'], axis=0, inplace=True)
        return super().export_pars(export_pars)
    
    def estimate_p(self):
        self.estimate_p_tissue_1scan()  

    def plot_fit(self, **kwargs):
        self.plot_fit_tissue_1scan(**kwargs)



class TwoScan(Model):            

    def predict_signal(self):
        R1 = self.predict_R1()
        p = self.p.value
        signal = dcmri.signalSPGRESS(p.TR, p.FA1, R1, p.S01k)
        k2 = np.nonzero(self.t() >= self.t0[1]-self.t0[0])[0]
        signal[k2] = dcmri.signalSPGRESS(p.TR, p.FA2, R1[k2], p.S02k)
        return signal
    
    def set_pars(self, TR, FA):
        self.p = self.pars_2scan(TR, FA) + super().pars()

    def export_pars(self, export_pars):
        export_pars.drop(['TR','FA1','FA2','S01k'],inplace=True)
        return super().export_pars(export_pars)

    def estimate_p(self):
        self.estimate_p_tissue_2scan()

    def plot_fit(self, **kwargs):
        self.plot_fit_tissue_2scan(**kwargs)


    