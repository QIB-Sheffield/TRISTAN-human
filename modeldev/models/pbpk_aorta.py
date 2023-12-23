import os
import math
import numpy as np
import matplotlib.pyplot as plt

import dcmri
import models.lib as lib

class Model(lib.FitFunc):
    def __init__(self,
            # Constants needed to predict pseudocontinuous signal
            weight = 70.0,              # Patient weight in kg
            agent = 'Primovist',
            conc = 0.25,                # mmol/mL (https://www.bayer.com/sites/default/files/2020-11/primovist-pm-en.pdf)
            dose = 0.025,               # mL per kg bodyweight (quarter dose)
            rate = 1,                   # Injection rate (mL/sec)
            dose_tolerance = 0.1,
            TR = 3.71/1000.0,           # Repetition time (sec)
            FA = 15.0,                  # Nominal flip angle (degrees)
            # Constants needed to predict pseudocontinuous signal
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
        self.weight = weight    
        self.conc = conc
        self.dose = dose
        self.rate = rate    
        self.agent = agent
        self.dose_tolerance = dose_tolerance
        self._set_dt()
        self.set_pars(TR, FA)
        self._set_df()
        
    def predict_R1(self, Ji):
        p = self.p.value
        t = self.t()
        # J_aorta = dcmri.prop_simple2_body(t, Ji,
        #     p.T_l, p.T_h,
        #     p.E_o, p.Tp_o, p.Te_o,
        #     p.E_b, p.T_v,
        #     tol=self.dose_tolerance)
        J_aorta = dcmri.prop_simple3_body(t, Ji,
            p.T_hl, p.D_hl,
            p.E_o, p.Tp_o, p.Te_o,
            p.E_b, 
            tol=self.dose_tolerance)
        self.cb = 1000*J_aorta/p.CO # mM
        # Return R
        rp = lib.rp(self.field_strength, self.agent)
        R1b = p.R10b + rp*self.cb
        return R1b
    
    def pars(self):
        return [
            # Aorta model
            ['R10b', "Baseline R1 (blood)", lib.R1_blood(), "1/sec", 0, np.inf, False, 4],
            ['CO', "Cardiac output", 100.0, "mL/sec", 0, np.inf, True, 3], # 6 L/min = 100 mL/sec
            # ['T_l', "Lung mean transit time", 6.0, "sec", 0, 30, True, 3],
            # ['T_h', "Heart mean transit time", 6.0, "sec", 0, 30, True, 3],
            # ['T_v', "Venous transit time", 5.0, "sec", 0, 30, True, 3],
            ['T_hl', "Heart-lung mean transit time", 10.0, "sec", 0, 30, True, 6],
            ['D_hl', "Heart-lung transit time dispersion", 20, "%", 5, 95, True, 6],
            ['Tp_o', "Organs mean transit time", 20.0, "sec", 0, 60, True, 6],
            ['E_o', "Extracellular extraction fraction", 0.15, "", 0, 0.50, True, 6],
            ['Te_o', "Extracellular mean transit time", 120.0, "sec", 0, 800.0, True, 6],
            ['E_b',"Body extraction fraction", 0.05,"", 0.01, 0.15, True, 6],
        ]
    
    def export_pars(self, export_pars):
        p = self.p.value
        # Convert to conventional units
        export_pars.loc['E_o', ['value', 'unit']] = [100*p.E_o, '%']
        export_pars.loc['E_b', ['value', 'unit']] = [100*p.E_b, '%']
        # Add derived parameters 
        #export_pars.loc['Tc'] = ["Mean circulation time", p.Tp_o+p.T_l+p.T_h, 'sec'] 
        export_pars.loc['Tc'] = ["Mean circulation time", p.Tp_o+p.T_hl, 'sec']         
        return export_pars
    
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

    def goodness(self):
        _, yref = self.xy_fitted() 
        self.predict_data()
        y = self.yp[self.valid][self.xind]
        ydist = np.linalg.norm(y - yref)
        loss = 100*ydist/np.linalg.norm(yref)
        return loss 
    
    def _fit_function(self, _, *params):
        self.it += 1 
        if self.callback:
            p = ' '.join([str(p) for p in params])
            print('Fitting ' + self.__class__.__name__ + ', iteration: ' + str(self.it))
            print('>> Parameter values: ' + p)
        self.p.loc[self.p.fit,'value'] = params
        self.predict_data() 
        y = self.yp[self.valid][self.xind]
        return y
    
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
        t = self.t()
        signal = self.predict_signal()
        x = self.tdce
        tacq = self.tdce[1]-self.tdce[0]
        self.yp = dcmri.sample(t, signal, x, tacq)
        return self.yp
    
    def t(self): 
        return np.arange(0, self.tmax+self.dt, self.dt)
    


    def plot_data_fit(self, ax1, xlim, legend):
        t = self.t()
        sig = self.predict_signal()
        tacq = self.tdce[1]-self.tdce[0]
        ax1.set_title('Signal')
        ax1.set(xlabel='Time (sec)', ylabel='MR Signal (a.u.)', xlim=xlim)
        xf = self.tdce[self.valid][self.xind]
        yf = self.Sdce[self.valid][self.xind]
        xi, yi = self.xy_ignored()
        ax1.plot(xf+tacq/2, yf,  'ro', label='Aorta data (fitted)')
        ax1.plot(xi+tacq/2, yi, marker='o', color='gray', label='Aorta data (ignored)', linestyle = 'None')
        ax1.plot(t, sig, 'b-', label=' Aorta fit' )
        if legend:
            ax1.legend()
 
    def plot_conc_fit(self, ax2, xlim, legend):
        t = self.t()
        ax2.set_title('Reconstructed concentration')
        ax2.set(xlabel='Time (sec)', ylabel='Concentration (mM)', xlim=xlim)
        ax2.plot(t, 0*t, color='black', label=self.plabel())
        ax2.plot(self.tdce, self.cb_direct(), 'ro', label='Direct inversion')
        ax2.plot(self.tdce, self.cb_direct_lin(), 'gx', label='Direct inversion (linear)')
        ax2.plot(t, self.cb, 'b-', label='Model-based')
        if legend:
            ax2.legend()

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


class OneScan(Model):

    def _set_dt(self):
        # Adjust internal time resolution
        duration = self.weight*self.dose/self.rate
        if duration != 0:
            if self.dt > duration/5:
                self.dt = duration/5  
        if self.tdce is not None:
            tacq = self.tdce[1]-self.tdce[0]
            self.tmax = np.amax(self.tdce) + tacq

    def predict_signal(self):
        t = self.t()
        p = self.p.value
        Ji = dcmri.injection(t, # mmol/sec
            self.weight, self.conc, self.dose, self.rate, p.BAT)
        R1b = self.predict_R1(Ji)
        Sb = dcmri.signalSPGRESS(p.TR, p.FA, R1b, p.S0b)
        return Sb

    def set_pars(self, TR, FA):
        self.p = [
            ['TR', "Repetition time", TR, "sec", 0, np.inf, False, 4],
            ['FA', "Flip angle", FA, "deg", 0, 180, False, 4],
            ['S0b', "Signal amplitude S0 in blood", 1200, "a.u.", 0, np.inf, False, 4],
            ['BAT', "Bolus arrival time", 60, "sec", 0, np.inf, True, 3],
        ] + self.pars()

    def export_pars(self, export_pars):
        export_pars.drop(['TR','FA','S0b'], axis=0, inplace=True)
        return super().export_pars(export_pars)
    
    def estimate_p(self):
        self.p.at['R10b','value'] = self.R1
        BAT = self.tdce[np.argmax(self.Sdce)]
        self.p.at['BAT','value'] = BAT 
        baseline = np.nonzero(self.tdce <= BAT-20)[0]
        n0 = baseline.size
        if n0 == 0: 
            n0 = 1
        p = self.p.value
        Srefb = dcmri.signalSPGRESS(p.TR, p.FA, p.R10b, 1)
        self.p.at['S0b','value'] = np.mean(self.Sdce[:n0]) / Srefb  

    def cb_direct(self):
        BAT = self.p.at['BAT','value']
        baseline = np.nonzero(self.tdce <= BAT-20)[0]
        n0 = baseline.size
        if n0 == 0: 
            n0 = 1
        S0 =  np.mean(self.Sdce[:n0])
        S = self.Sdce
        T10 = 1000/self.p.at['R10b','value']
        FA = self.p.at['FA','value']
        TR = 1000*self.p.at['TR','value']
        r1 = lib.rp(self.field_strength, self.agent)
        c =  dcmri.concentrationSPGRESS(S, S0, T10, FA, TR, r1)
        return c

    def cb_direct_lin(self):
        BAT = self.p.at['BAT','value']
        baseline = np.nonzero(self.tdce <= BAT-20)[0]
        n0 = baseline.size
        if n0 == 0: 
            n0 = 1
        S0 =  np.mean(self.Sdce[:n0])
        S = self.Sdce
        T10 = 1000/self.p.at['R10b','value']
        r1 = lib.rp(self.field_strength, self.agent)
        c =  dcmri.concentration_lin(S, S0, T10, r1)
        return c

    def plot_fit(self, show=True, save=False, path=None, prefix=''):
        self.plot_with_conc(show=show, save=save, path=path, prefix=prefix)
        BAT = self.p.value.BAT
        self.plot_with_conc(win='pass1', xrange=[BAT-20, BAT+160], show=show, save=save, path=path, prefix=prefix)



class TwoScan(Model):   

    def _set_dt(self):
        # Adjust internal time resolution
        duration1 = self.weight*self.dose[0]/self.rate
        duration2 = self.weight*self.dose[1]/self.rate
        duration = np.amin([duration1, duration2])
        if duration != 0:
            if self.dt > duration/5:
                self.dt = duration/5  
        if self.tdce is not None:
            tacq = self.tdce[1]-self.tdce[0]
            self.tmax = np.amax(self.tdce) + tacq   
    
    def predict_signal(self):
        t = self.t()
        p = self.p.value
        Ji = dcmri.injection(t, 
            self.weight, self.conc, self.dose[0], self.rate, p.BAT1, self.dose[1], p.BAT2)
        R1b = self.predict_R1(Ji)
        Sb = dcmri.signalSPGRESS(p.TR, p.FA1, R1b, p.S01b)
        k2 = np.nonzero(t >= self.t0[1]-self.t0[0])[0]
        Sb[k2] = dcmri.signalSPGRESS(p.TR, p.FA2, R1b[k2], p.S02b)
        return Sb
    
    def set_pars(self, TR, FA):
        self.p = [
            ['TR', "Repetition time", TR, "sec", 0, np.inf, False, 4],
            ['FA1', "Flip angle 1", FA, "deg", 0, 180, False, 4],
            ['FA2', "Flip angle 2", FA, "deg", 0, 180, False, 4],
            ['S01b', "Signal amplitude S0", 1200, "a.u.", 0, np.inf, False, 4],
            ['S02b', "Signal amplitude S0", 1200, "a.u.", 0, np.inf, True, 4],
            ['BAT1', "Bolus arrival time 1", 60, "sec", 0, np.inf, True, 3],
            ['BAT2', "Bolus arrival time 2", 60, "sec", 0, np.inf, True, 3],
        ] + self.pars() 

    def export_pars(self, export_pars):
        export_pars.drop(['TR','FA1','FA2','S01b'],inplace=True)
        return super().export_pars(export_pars)

    def estimate_p(self):
        self.p.at['R10b','value'] = self.R1[0]
        p = self.p.value

        k1 = self.tdce < self.t0[1]-self.t0[0]
        tdce1 = self.tdce[k1]
        Sdce1b = self.Sdce[k1]
        BAT1 = tdce1[np.argmax(Sdce1b)]
        self.p.at['BAT1','value'] = BAT1
        baseline = np.nonzero(tdce1 <= BAT1-20)[0]
        n0 = baseline.size
        if n0 == 0: 
            n0 = 1
        Srefb = dcmri.signalSPGRESS(p.TR, p.FA1, p.R10b, 1)
        self.p.at['S01b','value'] = np.mean(Sdce1b[:n0]) / Srefb
        
        k2 = self.tdce >= self.t0[1]-self.t0[0]
        tdce2 = self.tdce[k2]
        Sdce2b = self.Sdce[k2]
        BAT2 = tdce2[np.argmax(Sdce2b)]
        self.p.at['BAT2','value'] = BAT2
        n0 = math.floor(60/(tdce2[1]-tdce2[0])) 
        Srefb = dcmri.signalSPGRESS(p.TR, p.FA2, self.R1[1], 1)
        self.p.at['S02b','value'] = np.mean(Sdce2b[:n0]) / Srefb
        
    def plot_fit(self, show=True, save=False, path=None, prefix=''):
        self.plot_with_conc(show=show, save=save, path=path, prefix=prefix)
        BAT = self.p.value.BAT1
        self.plot_with_conc(win='shot1', xrange=[BAT-20, BAT+160], show=show, save=save, path=path, prefix=prefix)
        BAT = self.p.value.BAT2
        self.plot_with_conc(win='shot2', xrange=[BAT-20, BAT+160], show=show, save=save, path=path, prefix=prefix)

