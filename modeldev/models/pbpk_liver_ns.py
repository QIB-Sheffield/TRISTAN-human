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
            liver_volume = 1000, # mL
            # Constants used in model fitting
            BAT = None,  
        ):
        self.dt = dt
        self.tmax = tmax 
        self.field_strength = field_strength
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
        self.liver_volume = liver_volume
        # Constants needed for model fitting
        self.BAT = BAT
        self.set_pars(TR,FA)
        self._set_df()

    def predict_R1(self):
        p = self.p.value
        t = self.t()
        # Kve, E_l, FF_l, Kbh
        E_l = dcmri.lin(t, [p.E_li, p.E_lf])
        Kbh = dcmri.lin(t, [p.Kbh_i, p.Kbh_f])
        Ke = p.Kve/(1-E_l)
        J_gut = p.FF_l*self.J_aorta
        J_liver = dcmri.prop_comp(t, J_gut, p.T_g)
        Ne = dcmri.res_nscomp(t, J_liver, Ke)
        Nh = dcmri.res_nscomp(t, E_l*Ke*Ne, Kbh)
        self.Ce = 1000*Ne/self.liver_volume # mM
        self.Ch = 1000*Nh/self.liver_volume # mM
        # Return R
        rp = lib.rp(self.field_strength)
        rh = lib.rh(self.field_strength)
        R1l = p.R10l + rp*self.Ce + rh*self.Ch
        return R1l
    
    def pars(self):
        return [
            ['R10l', "Baseline R1 (liver)", lib.R1_liver(), "1/sec", 0, np.inf, False, 6],
            ['T_g', "Gut mean transit time", 8.0, "sec", 0, 30, True, 6],
            ['FF_l', "Liver flow fraction", 0.25, "", 0.01, 1, True, 6],
            ['Kve',"Liver venous excretion rate", 1/20,"1/sec", 1/60.0, 1/5.0, True, 6],
            ['E_li',"Liver extraction fraction (initial)", 0.07, "", 0.001, 0.2, True, 6],
            ['E_lf',"Liver extraction fraction (final)", 0.07, "", 0.001, 0.2, True, 6],
            ['Kbh_i',"Hepatocellular excretion rate (initial)", 1/(90*60), "1/sec", 1/(600*60), 1/(10*60), True, 6],
            ['Kbh_f',"Hepatocellular excretion rate (final)", 1/(90*60), "1/sec", 1/(600*60), 1/(10*60), True, 6],
        ]
    
    def export_pars(self, export_pars):
        p = self.p.value
        t = self.t()
        # Convert to conventional units
        export_pars.loc['E_li', ['value', 'unit']] = [100*p.E_li, '%']
        export_pars.loc['E_lf', ['value', 'unit']] = [100*p.E_lf, '%']
        # Add derived parameters 
        E_l = dcmri.lin(t, [p.E_li, p.E_lf])
        E_l = np.mean(E_l)
        Kbh = dcmri.lin(t, [p.Kbh_i, p.Kbh_f])
        Kbh = np.mean(Kbh)
        Ke = p.Kve/(1-E_l)
        Khe = E_l*Ke
        export_pars.loc['E_l'] = ["Liver extraction fraction", 100*E_l, '%'] 
        export_pars.loc['Khe'] = ["Hepatocellular tissue uptake rate", Khe, '1/sec'] 
        export_pars.loc['Kbh'] = ["Hepatocellular excretion rate", Kbh, '1/sec']         
        export_pars.loc['Th'] = ["Hepatocellular transit time", np.divide(1, Kbh)/60, 'min']
        export_pars.loc['Te'] = ["Extracellular transit time", np.divide(1, Ke), 'sec']
        return export_pars
    
    def plot_conc_fit(self, ax2, xlim, legend):
        t = self.t()
        ax2.set_title('Reconstructed concentration')
        ax2.set(xlabel='Time (sec)', ylabel='Tissue concentration (mM)', xlim=xlim)
        ax2.plot(t, 0*t, color='gray')
        ax2.plot(t, self.Ce, 'g-', label='Extracellular')
        ax2.plot(t, self.Ch, 'g--', label='Hepatocyte')
        ax2.plot(t, self.Ce+self.Ch, 'b-', label=self.plabel())
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
            ['S01l', "Signal amplitude S0", 1200, "a.u.", 0, np.inf, False, 4],
            ['S02l', "Signal amplitude S0", 1200, "a.u.", 0, np.inf, True, 4],
        ]
    
    def pars_1scan(self, TR, FA):
        return [
            ['TR', "Repetition time", TR, "sec", 0, np.inf, False, 4],
            ['FA', "Flip angle", FA, "deg", 0, 180, False, 4],
            ['S0l', "Signal amplitude S0", 1200, "a.u.", 0, np.inf, True, 4],
        ]
    
    def plot_fit_tissue_1scan(self, show=True, save=False, path=None, prefix=''):
        self.plot_with_conc(show=show, save=save, path=path, prefix=prefix)
        BAT = self.BAT
        self.plot_with_conc(win='win', xrange=[BAT-20, BAT+160], show=show, save=save, path=path, prefix=prefix)
        self.plot_with_conc(win='win_', xrange=[BAT-20, BAT+600], show=show, save=save, path=path, prefix=prefix)
        self.plot_with_conc(win='win__', xrange=[BAT-20, BAT+1200], show=show, save=save, path=path, prefix=prefix)

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
        self.p.at['R10l','value'] = self.R1
        p = self.p.value
        Sref = dcmri.signalSPGRESS(p.TR, p.FA, p.R10l, 1)
        baseline = np.nonzero(self.tdce <= self.BAT)[0]
        n0 = baseline.size
        if n0 == 0: 
            n0 = 1
        S0 = np.mean(self.Sdce[:n0]) / Sref
        self.p.at['S0l','value'] = S0

    def estimate_p_tissue_2scan(self):
        self.p.at['R10l','value'] = self.R1[0]
        p = self.p.value

        k1 = self.tdce < self.t0[1]-self.t0[0]
        tdce1 = self.tdce[k1]
        Sdce1 = self.Sdce[k1]
        BAT = self.BAT[0]
        Sref = dcmri.signalSPGRESS(p.TR, p.FA1, p.R10l, 1)
        baseline = np.nonzero(tdce1 <= BAT)[0]
        n0 = baseline.size
        if n0 == 0: 
            n0 = 1
        S01 = np.mean(Sdce1[:n0]) / Sref
        self.p.at['S01l','value'] = S01

        k2 = self.tdce >= self.t0[1]-self.t0[0]
        tdce2 = self.tdce[k2]
        Sdce2 = self.Sdce[k2]
        Sref = dcmri.signalSPGRESS(p.TR, p.FA2, self.R1[1], 1)
        n0 = math.floor(60/(tdce2[1]-tdce2[0])) # 1 minute baseline
        S02 = np.mean(Sdce2[:n0]) / Sref
        self.p.at['S02l','value'] = S02

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
        ax1.plot(xf+tacq/2, yf, 'ro', label='fitted data')
        ax1.plot(self.t(), self.predict_signal(), 'b-', label='fit' )
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
        signal = dcmri.signalSPGRESS(p.TR, p.FA, R1, p.S0l)
        return signal
    
    def set_pars(self, TR, FA):
        self.p = self.pars_1scan(TR, FA) + super().pars()

    def export_pars(self, export_pars):
        export_pars.drop(['TR','FA','S0l'], axis=0, inplace=True)
        return super().export_pars(export_pars)
    
    def estimate_p(self):
        self.estimate_p_tissue_1scan()  

    def plot_fit(self, **kwargs):
        self.plot_fit_tissue_1scan(**kwargs)



class TwoScan(Model):            

    def predict_signal(self):
        R1 = self.predict_R1()
        p = self.p.value
        signal = dcmri.signalSPGRESS(p.TR, p.FA1, R1, p.S01l)
        k2 = np.nonzero(self.t() >= self.t0[1]-self.t0[0])[0]
        signal[k2] = dcmri.signalSPGRESS(p.TR, p.FA2, R1[k2], p.S02l)
        return signal
    
    def set_pars(self, TR, FA):
        self.p = self.pars_2scan(TR, FA) + super().pars()

    def export_pars(self, export_pars):
        export_pars.drop(['TR','FA1','FA2','S01l'],inplace=True)
        return super().export_pars(export_pars)

    def estimate_p(self):
        self.estimate_p_tissue_2scan()

    def plot_fit(self, **kwargs):
        self.plot_fit_tissue_2scan(**kwargs)


    