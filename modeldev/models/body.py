import math
import numpy as np
import dcmri
import models.lib as lib

class Aorta(lib.SuperModel):
    def __init__(self,
            # Constants needed to predict pseudocontinuous signal
            weight = 70.0,              # Patient weight in kg
            conc = 0.25,                # mmol/mL (https://www.bayer.com/sites/default/files/2020-11/primovist-pm-en.pdf)
            dose = 0.025,               # mL per kg bodyweight (quarter dose)
            rate = 1,                   # Injection rate (mL/sec)
            dose_tolerance = 0.1,
            TR = 3.71/1000.0,           # Repetition time (sec)
            FA = 15.0,                  # Nominal flip angle (degrees)
            # Constants used in model fitting
            **kwargs,
            ):
        super().__init__(**kwargs)
        # Essential constants
        self.weight = weight           
        self.conc = conc
        self.dose = dose
        self.rate = rate    
        self.dose_tolerance = dose_tolerance
        self.set_pars(TR, FA)
        self._set_df()
        self._set_dt()

    def predict_R1(self, Ji):
        p = self.p.value
        t = self.t()
        # Derived constants
        vp = (1-self.Hct)*p.vb  # Kidney plasma volume
        Fp = vp/p.Tp/(1+p.FF)   # Kidney plasma flow
        Ft = p.FF*Fp            # Kidney Tubular flow
        Ek = Ft/(Ft+Fp)         # Kidney extraction fraction
        k_he = dcmri.lin(t, [p.k_he_i, p.k_he_f])
        Kbh = dcmri.lin(t, [p.Kbh_i, p.Kbh_f])
        Khe = np.mean(k_he)/p.ve
        Ee = 0
        # Generate aorta conc
        _, Jb = dcmri.propagate_simple_body(t, Ji, 
            p.MTThl, p.El, p.MTTe, p.MTTo, p.TTDo, Ee, tol=self.dose_tolerance)
        self.cb = Jb*1000/p.CO  # (mM)
        # Generate liver conc
        ca = dcmri.propagate_dd(t, self.cb, p.Tdel, p.Te)
        self.Ce = p.ve*ca/(1-self.Hct)
        self.Ch = dcmri.res_nscomp(t, k_he*ca, Kbh) 
        # Generate kidney conc
        ca = self.cb/(1-self.Hct)
        self.Cp = dcmri.res_comp(t, Fp*ca, p.Tp)
        cp = self.Cp/vp
        self.Ct = dcmri.res_trap(t, Ft*cp)
        # Return R1
        rp = lib.rp(self.field_strength)
        rh = lib.rh(self.field_strength)    
        R1a = p.R10b + rp*self.cb
        R1l = p.R10l + rp*self.Ce + rh*self.Ch
        R1k = p.R10k + rp*(self.Cp + self.Ct)
        return R1a, R1l, R1k
    
    def pars(self):
        return [
            # Aorta parameters
            ['R10b', "Baseline blood R1", lib.R1_blood(), "1/sec", 0, np.inf, False, 4],
            ['CO', "Cardiac output", 100.0, "mL/sec", 0, np.inf, True, 3], # 6 L/min = 100 mL/sec
            ['MTThl', "Heart & lung mean transit time", 8.0, "sec", 0, 30, True, 3],
            ['MTTo', "Other organs mean transit time", 20.0, "sec", 0, 60, True, 3],
            ['TTDo', "Other organs transit time dispersion", 10.0, "sec", 0, np.inf, True, 2],
            ['MTTe', "Storage compartment mean transit time", 120.0, "sec", 0, 800.0, True, 3],
            ['El', "Storage compartment leakage fraction", 0.15, "", 0, 0.50, True, 4],
            # Liver parameters
            ['R10l', "Baseline liver R1", lib.R1_liver(), "1/sec", 0, np.inf, False, 4],
            ['Tdel', "Gut delay time", 5.0, 'sec', 0, 20.0, True, 6],  
            ['Te', "Extracellular transit time", 30.0, 'sec', 0, 60, True, 6],
            ['ve', "Extracellular volume fraction", 0.3, 'mL/mL', 0.01, 0.6, True, 6],
            ['k_he_i', "Hepatocellular uptake rate", 20/6000, 'mL/sec/mL', 0, np.inf, True, 6],
            ['k_he_f', "Hepatocellular uptake rate", 20/6000, 'mL/sec/mL', 0, np.inf, True, 6],
            ['Kbh_i', "Biliary tissue excretion rate", 1/(30*60), 'mL/sec/mL', 1/(2*60*60), 1/(10*60), True, 6],
            ['Kbh_f', "Biliary tissue excretion rate", 1/(30*60), 'mL/sec/mL', 1/(2*60*60), 1/(10*60), True, 6],
            # Kidney parameters
            ['R10k', "Baseline kidney R1", lib.R1_kidney(), "1/sec", 0, np.inf, False, 4], 
            ['vb', "Blood volume", 0.3, 'mL/mL', 0.01, 0.99, True, 6],
            ['Tp', "Plasma transit time", 5, 'sec', 0, 15, True, 6],
            ['FF', "Filtration Fraction", 0.1, '', 0.01, 0.99, True, 6],
        ]
 
    def plot_conc_fit(self, ax2, xlim, legend):
        t = self.t()
        ax2.set_title('Reconstructed concentration')
        ax2.set(xlabel='Time (sec)', ylabel='Concentration (mM)', xlim=xlim)
        ax2.plot(t, 0*t, color='black')
        ax2.plot(t, self.cb, 'b-', label=self.plabel())
        if legend:
            ax2.legend()


class AortaOneScan(Aorta):

    def _set_dt(self):
        # Adjust internal time resolution
        duration = self.weight*self.dose/self.rate
        if duration != 0:
            if self.dt > duration/5:
                self.dt = duration/5  
        if self.tdce is not None:
            tacq = self.tdce[1]-self.tdce[0]
            self.tmax = np.amax([np.amax(self.tdce), np.amax(self.tR1)]) + tacq

    def predict_signal(self):
        p = self.p.value
        t = self.t()
        Ji = dcmri.injection(t, 
            self.weight, self.conc, self.dose, self.rate, p.BAT)
        R1 = self.predict_R1(Ji)
        signal = dcmri.signalSPGRESS(p.TR, p.FA, R1, p.S0)
        return signal
    
    def set_pars(self, TR, FA):
        self.p = self.pars_1scan(TR, FA) + [
            ['BAT', "Bolus arrival time", 60, "sec", 0, np.inf, True, 3],
        ]  + self.pars()

    def export_pars(self, export_pars):
        export_pars.drop(['TR','FA','S0'], axis=0, inplace=True)
        return export_pars
    
    def estimate_p(self):
        self.p.at['R10','value'] = self.vR1[0]
        p = self.p.value
        BAT = self.tdce[np.argmax(self.Sdce)]
        baseline = np.nonzero(self.tdce <= BAT-20)[0]
        n0 = baseline.size
        if n0 == 0: 
            n0 = 1
        Sref = dcmri.signalSPGRESS(p.TR, p.FA, p.R10, 1)
        S0 = np.mean(self.Sdce[:n0]) / Sref
        self.p.at['S0','value'] = S0
        self.p.at['BAT','value'] = BAT  

    def plot_fit(self, show=True, save=False, path=None, prefix=''):
        self.plot_with_conc(show=show, save=save, path=path, prefix=prefix)
        BAT = self.p.value.BAT
        self.plot_with_conc(win='pass1', xrange=[BAT-20, BAT+160], show=show, save=save, path=path, prefix=prefix)


class AortaTwoScan(Aorta):   

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
            self.tmax = np.amax([np.amax(self.tdce), np.amax(self.tR1)]) + tacq     
  
    def predict_signal(self):
        p = self.p.value
        t = self.t()
        Ji = dcmri.injection(t, 
            self.weight, self.conc, self.dose[0], self.rate, p.BAT1, self.dose[1], p.BAT2)
        R1 = self.predict_R1(Ji)
        signal = dcmri.signalSPGRESS(p.TR, p.FA1, R1, p.S01)
        k2 = np.nonzero(self.t() >= self.t0[1]-self.t0[0])[0]
        signal[k2] = dcmri.signalSPGRESS(p.TR, p.FA2, R1[k2], p.S02)
        return signal

    def set_pars(self, TR, FA):
        self.p = self.pars_2scan(TR, FA) + [
            ['BAT1', "Bolus arrival time 1", 60, "sec", 0, np.inf, True, 3],
            ['BAT2', "Bolus arrival time 2", 60, "sec", 0, np.inf, True, 3],
        ]  + self.pars() 

    def export_pars(self, export_pars):
        export_pars.drop(['TR','FA1','FA2','S01'],inplace=True)
        return export_pars

    def estimate_p(self):
        self.p.at['R10','value'] = self.vR1[0]
        p = self.p.value

        k1 = self.tdce < self.t0[1]-self.t0[0]
        tdce1 = self.tdce[k1]
        Sdce1 = self.Sdce[k1]
        BAT1 = tdce1[np.argmax(Sdce1)]
        Sref = dcmri.signalSPGRESS(p.TR, p.FA1, p.R10, 1)
        baseline = np.nonzero(tdce1 <= BAT1-20)[0]
        n0 = baseline.size
        if n0 == 0: 
            n0 = 1
        S01 = np.mean(Sdce1[:n0]) / Sref
        self.p.at['S01','value'] = S01
        self.p.at['BAT1','value'] = BAT1

        k2 = self.tdce >= self.t0[1]-self.t0[0]
        tdce2 = self.tdce[k2]
        Sdce2 = self.Sdce[k2]
        Sref = dcmri.signalSPGRESS(p.TR, p.FA2, self.vR1[2], 1)
        n0 = math.floor(60/(tdce2[1]-tdce2[0])) # 1 minute baseline
        BAT2 = tdce2[np.argmax(Sdce2)]
        S02 = np.mean(Sdce2[:n0]) / Sref
        self.p.at['S02','value'] = S02
        self.p.at['BAT2','value'] = BAT2

    def plot_fit(self, show=True, save=False, path=None, prefix=''):
        self.plot_with_conc(show=show, save=save, path=path, prefix=prefix)
        BAT = self.p.value.BAT1
        self.plot_with_conc(win='shot1', xrange=[BAT-20, BAT+160], show=show, save=save, path=path, prefix=prefix)
        BAT = self.p.value.BAT2
        self.plot_with_conc(win='shot2', xrange=[BAT-20, BAT+160], show=show, save=save, path=path, prefix=prefix)

