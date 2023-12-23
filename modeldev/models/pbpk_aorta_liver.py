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
            conc = 0.25,                # mmol/mL (https://www.bayer.com/sites/default/files/2020-11/primovist-pm-en.pdf)
            dose = 0.025,               # mL per kg bodyweight (quarter dose)
            rate = 1,                   # Injection rate (mL/sec)
            dose_tolerance = 0.1,
            TR = 3.71/1000.0,           # Repetition time (sec)
            FA = 15.0,                  # Nominal flip angle (degrees)
            liver_volume = 1000, # mL
            Hct = 0.45,
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
            # Initial values
            aorta = None,
            liver = None,
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
        self.dose_tolerance = dose_tolerance
        self.liver_volume = liver_volume
        self.Hct = Hct
        self._set_dt()
        self.set_pars(TR, FA)
        self._set_df()
        if aorta is not None:
            for var in aorta.index:
                if var in self.p.index:
                    self.p.at[var,'value'] = aorta.at[var,'value']
        if liver is not None:
            for var in liver.index:
                if var in self.p.index:
                    self.p.at[var,'value'] = liver.at[var,'value']
        self.set_R10()


    def predict_R1(self, Ji):
        p = self.p.value
        t = self.t()
        rp = lib.rp(self.field_strength)
        rh = lib.rh(self.field_strength)
        # Aorta
        J_aorta = dcmri.prop_body2_liver(t, Ji,
            p.T_hl, p.D_hl,
            p.E_o, p.Tp_o, p.Te_o,
            p.T_g, p.FF_l, p.E_l, p.Te,
            p.E_k, 
            tol=self.dose_tolerance)
        self.cb = 1000*J_aorta/p.CO # mM
        R1b = p.R10b + rp*self.cb
        # Liver
        J_liver = p.FF_l*J_aorta
        J_liver = dcmri.prop_comp(t, J_liver, p.T_g)
        Ne, Nh = dcmri.res_liver(t, J_liver, 1/p.Te, 0, 1/p.Th, p.E_l)
        self.Ce_l = 1000*Ne/self.liver_volume # mM
        self.Ch_l = 1000*Nh/self.liver_volume # mM
        R1l = p.R10l + rp*self.Ce_l + rh*self.Ch_l  
        return R1b, R1l
    
    
    def pars(self):
        return [
            # Aorta model
            ['R10b', "Baseline R1 (blood)", lib.R1_blood(), "1/sec", 0, np.inf, False, 4],
            ['CO', "Cardiac output", 100.0, "mL/sec", 0, np.inf, True, 3], # 6 L/min = 100 mL/sec
            # ['T_lh', "Heart & lung mean transit time", 8.0, "sec", 0, 30, True, 3],
            ['T_hl', "Heart-lung mean transit time", 10.0, "sec", 0, 30, False, 6],
            ['D_hl', "Heart-lung transit time dispersion", 20, "%", 0, 100, False, 6],
            ['Tp_o', "Organs mean transit time", 20.0, "sec", 0, 60, True, 3],
            ['E_o', "Extracellular extraction fraction", 0.15, "", 0, 0.50, True, 4],
            ['Te_o', "Extracellular mean transit time", 120.0, "sec", 0, 800.0, True, 3],
            # Liver model
            ['R10l', "Baseline R1 (liver)", lib.R1_liver(), "1/sec", 0, np.inf, False, 6],
            ['FF_l', "Liver flow fraction", 0.25, "", 0.01, 1, True, 6],
            #['T_ld', "Liver delay time", 2.0, "sec", 0, 20, True, 6],
            ['T_g', "Gut mean transit time", 10.0, "sec", 1, 20, True, 6],
            #['D_g', "Gut dispersion", 50.0, "%", 10, 90, True, 6],
            ['Te',"Extracellular transit time", 30,"sec", 15, 60.0, True, 6],
            ['E_l',"Liver extraction fraction", 0.04, "", 0.00, 0.10, True, 6],
            ['Th',"Hepatocellular transit time", 30*60, "sec", 10*60, 600*60, True, 6],
            # Kidney model
            ['E_k',"Kidney extraction fraction", 0.15, "", 0.001, 0.2, True, 6],
        ]
    
    def export_pars(self, export_pars):
        p = self.p.value
        # Aorta
        # Convert to conventional units
        export_pars.loc['E_o', ['value', 'unit']] = [100*p.E_o, '%']
        # Add derived parameters 
        export_pars.loc['Tc'] = ["Mean circulation time", p.Tp_o+p.T_hl, 'sec'] 
        # Liver
        # Convert to conventional units
        export_pars.loc['FF_l', ['value', 'unit']] = [100*p.FF_l, '%']
        export_pars.loc['E_l', ['value', 'unit']] = [100*p.E_l, '%']
        export_pars.loc['Th', ['value', 'unit']] = [p.Th/60, 'min']
        # Add derived parameters 
        Kve = 1/p.Te
        Ke = Kve/(1-p.E_l)
        Khe = p.E_l*Ke
        lev = p.FF_l*p.CO*p.Te/self.liver_volume
        Kbh = 1/p.Th
        export_pars.loc['Khe'] = ["Hepatocellular tissue uptake rate", 6000*Khe, 'mL/min/100mL']         
        export_pars.loc['Kbh'] = ["Hepatocellular excretion rate", 6000*Kbh, 'mL/min/100mL']
        # Kidney   
        # Convert to conventional units
        export_pars.loc['E_k', ['value', 'unit']] = [100*p.E_k, '%']
        # Add derived
        export_pars.loc['FF'] = ["Filtration fraction", 100*np.divide(p.E_k, 1-p.E_k), '%']    
        # Mixed derived (liver)
        export_pars.loc['LBF'] = ["Liver blood flow", 60*p.FF_l*p.CO, 'mL/min']
        export_pars.loc['LP'] = ["Liver perfusion", 6000*p.FF_l*p.CO/self.liver_volume, 'mL/min/100mL']
        export_pars.loc['LEHV'] = ["Liver extra-hepatocellular volume", 100*lev, 'mL/100mL']
        export_pars.loc['CL_l'] = ['Liver blood clearance', 60*p.E_l*p.FF_l*p.CO, 'mL/min']
        export_pars.loc['E_lb'] = ['Liver whole body extraction fraction', 100*p.E_l*p.FF_l, '%']
        export_pars.loc['k_he'] = ["Hepatocellular tissue uptake rate", 6000*Khe*lev, 'mL/min/100mL'] 
        export_pars.loc['k_bh'] = ["Hepatocellular excretion rate", 6000*Kbh*(1-lev), 'mL/min/100mL']          
        # Mixed derived (body)
        export_pars.loc['E_b'] = ['Whole body extraction fraction', 100*(p.E_l*p.FF_l+p.E_k*(1-p.FF_l)), '%']
        export_pars.loc['C_l'] = ['Liver contribution', 100*p.E_l*p.FF_l/(p.E_l*p.FF_l+p.E_k*(1-p.FF_l)), '%']
        export_pars.loc['C_k'] = ['Kidney contribution', 100*p.E_k*(1-p.FF_l)/(p.E_l*p.FF_l+p.E_k*(1-p.FF_l)), '%']
        return export_pars
    
    def get_valid(self, valid, tstart, tstop):
        if valid is None:
            valid = np.full(self.tdce.shape, True)
        if tstart is None:
            tstart = np.amin(self.tdce)
        if tstop is None:
            tstop = np.amax(self.tdce)
        valid = np.where(valid)
        t = self.tdce[valid]
        return valid, np.nonzero((t>=tstart) & (t<=tstop))[0]

    def _set_xind(self, valid, tstart, tstop):
        if tstart is None:
            tstart = 3*[np.amin(self.tdce)]
        if tstop is None:
            tstop = 3*[np.amax(self.tdce)]
        vb, xb = self.get_valid(valid[0], tstart[0], tstop[0])
        vl, xl = self.get_valid(valid[1], tstart[1], tstop[1])
        self.valid = [vb, vl]
        self.xind = [xb, xl]

    def xy_fitted(self):
        x = np.concatenate([
            self.tdce[self.valid[0]][self.xind[0]], 
            self.tdce[self.valid[1]][self.xind[1]], 
            ])
        y = np.concatenate([
            self.Sdce[0][self.valid[0]][self.xind[0]], 
            self.Sdce[1][self.valid[1]][self.xind[1]], 
            ])
        return x, y 
    
    def xy_ignored(self, i):
        valid = np.full(self.tdce.shape, False)
        valid[self.valid[i][0][self.xind[i]]] = True
        ind = np.where(valid==False)
        return self.tdce[ind], self.Sdce[i][ind]
    
    def predict_data(self):
        t = self.t()
        Sb, Sl = self.predict_signal()
        tacq = self.tdce[1]-self.tdce[0]
        Sb = dcmri.sample(t, Sb, self.tdce, tacq)
        Sl = dcmri.sample(t, Sl, self.tdce, tacq)
        return Sb, Sl

    def goodness(self):
        _, yref = self.xy_fitted() 
        Sb, Sl = self.predict_data()
        y = np.concatenate([
            Sb[self.valid[0]][self.xind[0]], 
            Sl[self.valid[1]][self.xind[1]], 
            ])
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
        Sb, Sl = self.predict_data() 
        return np.concatenate([
            Sb[self.valid[0]][self.xind[0]], 
            Sl[self.valid[1]][self.xind[1]], 
            ])
    
    def t(self): 
        return np.arange(0, self.tmax+self.dt, self.dt)

    def plot_data_fit(self, ax1, xlim, legend):
        t = self.t()
        Sb, Sl = self.predict_signal()
        sig = np.concatenate([Sb, Sl])
        tacq = self.tdce[1]-self.tdce[0]
        ax1.set_title('Signal')
        ax1.set(xlabel='Time (sec)', ylabel='MR Signal (a.u.)', xlim=xlim)
        i=0
        xf = self.tdce[self.valid[i]][self.xind[i]]
        yf = self.Sdce[i][self.valid[i]][self.xind[i]]
        xi, yi = self.xy_ignored(i)
        ax1.plot(xf+tacq/2, yf, 'ro', label='Aorta data (fitted)')
        ax1.plot(xi+tacq/2, yi, marker='o', color='gray', label='Aorta data (ignored)', linestyle = 'None')
        ax1.plot(t, sig[:t.size], 'r-', label=' Aorta fit' )
        i=1
        xf = self.tdce[self.valid[i]][self.xind[i]]
        yf = self.Sdce[i][self.valid[i]][self.xind[i]]
        xi, yi = self.xy_ignored(i)
        ax1.plot(xf+tacq/2, yf, 'bo', label='Liver data (fitted)')
        ax1.plot(xi+tacq/2, yi, marker='o', color='gray', label='Liver data (ignored)', linestyle = 'None')
        ax1.plot(t, sig[t.size:2*t.size], 'b-', label=' Liver fit' )
        if legend:
            ax1.legend()
 
    def plot_conc_fit(self, ax2, xlim, legend):
        t = self.t()
        ax2.set_title('Reconstructed concentration')
        ax2.set(xlabel='Time (sec)', ylabel='Concentration (mM)', xlim=xlim)
        ax2.plot(t, 0*t, color='black', label=self.plabel())
        ax2.plot(t, self.cb, 'r-', label='Aorta')
        ax2.plot(t, self.Ce_l + self.Ch_l, 'b-', label='Liver')
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
        R1b, R1l = self.predict_R1(Ji)
        Sb = dcmri.signalSPGRESS(p.TR, p.FA, R1b, p.S0b)
        Sl = dcmri.signalSPGRESS(p.TR, p.FA, R1l, p.S0l)
        return Sb, Sl

    def set_pars(self, TR, FA, *args):
        self.p = [
            ['TR', "Repetition time", TR, "sec", 0, np.inf, False, 4],
            ['FA', "Flip angle", FA, "deg", 0, 180, False, 4],
            ['S0b', "Signal amplitude S0 in blood", 1200, "a.u.", 0, np.inf, False, 4],
            ['S0l', "Signal amplitude S0 in liver", 1200, "a.u.", 0, np.inf, False, 4],
            ['BAT', "Bolus arrival time", 60, "sec", 0, np.inf, False, 3],
        ] + self.pars(*args)

    def export_pars(self, export_pars):
        export_pars.drop(['TR','FA','S0b','S0l'], axis=0, inplace=True)
        return super().export_pars(export_pars)
    
    def set_R10(self):
        self.p.at['R10b','value'] = self.R1[0]
        self.p.at['R10l','value'] = self.R1[1]
    
    def estimate_p(self):
        BAT = self.tdce[np.argmax(self.Sdce[0])]
        self.p.at['BAT','value'] = BAT 
        baseline = np.nonzero(self.tdce <= BAT-20)[0]
        n0 = baseline.size
        if n0 == 0: 
            n0 = 1
        p = self.p.value
        Srefb = dcmri.signalSPGRESS(p.TR, p.FA, p.R10b, 1)
        Srefl = dcmri.signalSPGRESS(p.TR, p.FA, p.R10l, 1)
        self.p.at['S0b','value'] = np.mean(self.Sdce[0][:n0]) / Srefb
        self.p.at['S0l','value'] = np.mean(self.Sdce[1][:n0]) / Srefl

    def estimate_S0(self):
        BAT = self.p.at['BAT','value']
        baseline = np.nonzero(self.tdce <= BAT-20)[0]
        n0 = baseline.size
        if n0 == 0: 
            n0 = 1
        p = self.p.value
        Srefb = dcmri.signalSPGRESS(p.TR, p.FA, p.R10b, 1)
        Srefl = dcmri.signalSPGRESS(p.TR, p.FA, p.R10l, 1)
        self.p.at['S0b','value'] = np.mean(self.Sdce[0][:n0]) / Srefb
        self.p.at['S0l','value'] = np.mean(self.Sdce[1][:n0]) / Srefl

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
        R1b, R1l = self.predict_R1(Ji)
        Sb = dcmri.signalSPGRESS(p.TR, p.FA1, R1b, p.S01b)
        Sl = dcmri.signalSPGRESS(p.TR, p.FA1, R1l, p.S01l)
        k2 = np.nonzero(t >= self.t0[1]-self.t0[0])[0]
        Sb[k2] = dcmri.signalSPGRESS(p.TR, p.FA2, R1b[k2], p.S02b)
        Sl[k2] = dcmri.signalSPGRESS(p.TR, p.FA2, R1l[k2], p.S02l)
        return Sb, Sl
    
    def set_pars(self, TR, FA, *args):
        self.p = [
            ['TR', "Repetition time", TR, "sec", 0, np.inf, False, 4],
            ['FA1', "Flip angle 1", FA, "deg", 0, 180, False, 4],
            ['FA2', "Flip angle 2", FA, "deg", 0, 180, False, 4],
            ['S01b', "Signal amplitude S0", 1200, "a.u.", 0, np.inf, False, 4],
            ['S02b', "Signal amplitude S0", 1200, "a.u.", 0, np.inf, True, 4],
            ['S01l', "Signal amplitude S0", 1200, "a.u.", 0, np.inf, False, 4],
            ['S02l', "Signal amplitude S0", 1200, "a.u.", 0, np.inf, True, 4],
            ['BAT1', "Bolus arrival time 1", 60, "sec", 0, np.inf, False, 3],
            ['BAT2', "Bolus arrival time 2", 60, "sec", 0, np.inf, False, 3],
        ] + self.pars(*args) 

    def export_pars(self, export_pars):
        export_pars.drop(['TR','FA1','FA2','S01b','S01l'],inplace=True)
        return super().export_pars(export_pars)
    
    def set_R10(self):
        self.p.at['R10b','value'] = self.R1[0][0]
        self.p.at['R10l','value'] = self.R1[1][0]

    def estimate_p(self):
        p = self.p.value

        k1 = self.tdce < self.t0[1]-self.t0[0]
        tdce1 = self.tdce[k1]
        Sdce1b = self.Sdce[0][k1]
        Sdce1l = self.Sdce[1][k1]
        BAT1 = tdce1[np.argmax(Sdce1b)]
        self.p.at['BAT1','value'] = BAT1
        baseline = np.nonzero(tdce1 <= BAT1-20)[0]
        n0 = baseline.size
        if n0 == 0: 
            n0 = 1
        Srefb = dcmri.signalSPGRESS(p.TR, p.FA1, p.R10b, 1)
        Srefl = dcmri.signalSPGRESS(p.TR, p.FA1, p.R10l, 1)
        self.p.at['S01b','value'] = np.mean(Sdce1b[:n0]) / Srefb
        self.p.at['S01l','value'] = np.mean(Sdce1l[:n0]) / Srefl
        
        k2 = self.tdce >= self.t0[1]-self.t0[0]
        tdce2 = self.tdce[k2]
        Sdce2b = self.Sdce[0][k2]
        Sdce2l = self.Sdce[1][k2]
        BAT2 = tdce2[np.argmax(Sdce2b)]
        self.p.at['BAT2','value'] = BAT2
        n0 = math.floor(60/(tdce2[1]-tdce2[0])) 
        Srefb = dcmri.signalSPGRESS(p.TR, p.FA2, self.R1[0][1], 1)
        Srefl = dcmri.signalSPGRESS(p.TR, p.FA2, self.R1[1][1], 1)
        self.p.at['S02b','value'] = np.mean(Sdce2b[:n0]) / Srefb
        self.p.at['S02l','value'] = np.mean(Sdce2l[:n0]) / Srefl

    def estimate_S0(self):
        p = self.p.value

        k1 = self.tdce < self.t0[1]-self.t0[0]
        tdce1 = self.tdce[k1]
        Sdce1b = self.Sdce[0][k1]
        Sdce1l = self.Sdce[1][k1]
        BAT1 = self.p.at['BAT1','value']
        baseline = np.nonzero(tdce1 <= BAT1-20)[0]
        n0 = baseline.size
        if n0 == 0: 
            n0 = 1
        Srefb = dcmri.signalSPGRESS(p.TR, p.FA1, p.R10b, 1)
        Srefl = dcmri.signalSPGRESS(p.TR, p.FA1, p.R10l, 1)
        self.p.at['S01b','value'] = np.mean(Sdce1b[:n0]) / Srefb
        self.p.at['S01l','value'] = np.mean(Sdce1l[:n0]) / Srefl
        
        k2 = self.tdce >= self.t0[1]-self.t0[0]
        tdce2 = self.tdce[k2]
        Sdce2b = self.Sdce[0][k2]
        Sdce2l = self.Sdce[1][k2]
        n0 = math.floor(60/(tdce2[1]-tdce2[0])) 
        Srefb = dcmri.signalSPGRESS(p.TR, p.FA2, self.R1[0][1], 1)
        Srefl = dcmri.signalSPGRESS(p.TR, p.FA2, self.R1[1][1], 1)
        self.p.at['S02b','value'] = np.mean(Sdce2b[:n0]) / Srefb
        self.p.at['S02l','value'] = np.mean(Sdce2l[:n0]) / Srefl

    def plot_fit(self, show=True, save=False, path=None, prefix=''):
        self.plot_with_conc(show=show, save=save, path=path, prefix=prefix)
        BAT = self.p.value.BAT1
        self.plot_with_conc(win='shot1', xrange=[BAT-20, BAT+160], show=show, save=save, path=path, prefix=prefix)
        BAT = self.p.value.BAT2
        self.plot_with_conc(win='shot2', xrange=[BAT-20, BAT+160], show=show, save=save, path=path, prefix=prefix)

