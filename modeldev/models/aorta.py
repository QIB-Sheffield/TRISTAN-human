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

    def plot_data_fit(self, ax1, xlim, legend):
        xf, yf = self.xy_fitted()
        xi, yi = self.xy_ignored()
        tacq = self.tdce[1]-self.tdce[0]
        ax1.set(xlabel='Time (min)', ylabel='MR Signal (a.u.)', xlim=np.array(xlim)/60)
        ax1.plot((xi+tacq/2)/60, yi, marker='o', color='gray', label='ignored data', linestyle = 'None')
        ax1.plot((xf+tacq/2)/60, yf, marker='o', color='lightcoral', label='fitted data', linestyle = 'None')
        ax1.plot(self.t()/60, self.predict_signal(), linestyle='-', color='darkred', linewidth=3.0, label='fit' )
        ax1.plot(np.array([self.tdce[0]]+self.tR1[1:])/60, self.s_molli(), color='black', marker='x', linestyle='None', label='MOLLI')
        if legend:
            ax1.legend()

    def plot_conc_fit(self, ax2, xlim, legend):
        t = self.t()/60
        ax2.set(xlabel='Time (min)', ylabel='Concentration (mM)', xlim=np.array(xlim)/60)
        ax2.plot(t, 0*t, color='gray')
        ax2.plot(t, self.cb, linestyle='-', color='darkred', linewidth=3.0, label='Blood')
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

    def predict_R1(self):
        p = self.p.value
        t = self.t()
        Ji = dcmri.injection(t, 
            self.weight, self.conc, self.dose, self.rate, p.BAT)
        _, Jb = dcmri.propagate_simple_body(t, Ji, 
            p.MTThl, p.El, p.MTTe, p.MTTo, p.TTDo, p.Ee, tol=self.dose_tolerance)
        self.cb = Jb*1000/p.CO  # (mM)
        rp = lib.rp(self.field_strength)
        R1 = p.R10 + rp * self.cb
        return R1
    
    def pars(self):
        return [
            ['R10', "Baseline R1", lib.R1_blood(), "1/sec", 0, np.inf, False, 4],
            ['BAT', "Bolus arrival time", 60, "sec", 0, np.inf, True, 3],
            ['CO', "Cardiac output", 100.0, "mL/sec", 0, np.inf, True, 3], # 6 L/min = 100 mL/sec
            ['MTThl', "Heart & lung mean transit time", 8.0, "sec", 0, 30, True, 3],
            ['MTTo', "Other organs mean transit time", 20.0, "sec", 0, 60, True, 3],
            ['TTDo', "Other organs transit time dispersion", 10.0, "sec", 0, np.inf, True, 2],
            ['MTTe', "Storage compartment mean transit time", 120.0, "sec", 0, 800.0, True, 3],
            ['El', "Storage compartment leakage fraction", 0.15, "", 0, 0.50, True, 4],
            ['Ee',"Kidney & Liver Extraction fraction", 0.05,"", 0.01, 0.15, True, 4],
        ]

    def predict_signal(self):
        R1 = self.predict_R1()
        p = self.p.value
        signal = dcmri.signalSPGRESS(p.TR, p.FA, R1, p.S0)
        return signal
    
    def set_pars(self, TR, FA):
        self.p = self.pars_1scan(TR, FA) + self.pars()

    def export_pars(self, export_pars):
        p = self.p.value
        export_pars.drop(['TR','FA','S0'], axis=0, inplace=True)
        export_pars.loc['El', ['value', 'unit']] = [100*p.El, '%']
        export_pars.loc['Ee', ['value', 'unit']] = [100*p.Ee, '%']
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

    def predict_R1(self):
        p = self.p.value
        t = self.t()
        Ji = dcmri.injection(t, 
            self.weight, self.conc, self.dose[0], self.rate, p.BAT1, self.dose[1], p.BAT2)
        _, Jb = dcmri.propagate_simple_body(t, Ji, 
            p.MTThl, p.El, p.MTTe, p.MTTo, p.TTDo, p.Ee, tol=self.dose_tolerance)
        self.cb = Jb*1000/p.CO  # (mM)
        rp = lib.rp(self.field_strength)
        R1 = p.R10 + rp * self.cb
        return R1
    
    def pars(self):
        return [
            ['R10', "Baseline R1", lib.R1_blood(), "1/sec", 0, np.inf, False, 4],
            ['BAT1', "Bolus arrival time 1", 60, "sec", 0, np.inf, True, 3],
            ['BAT2', "Bolus arrival time 2", 60, "sec", 0, np.inf, True, 3],
            ['CO', "Cardiac output", 100.0, "mL/sec", 0, np.inf, True, 3], # 6 L/min = 100 mL/sec
            ['MTThl', "Heart & lung mean transit time", 8.0, "sec", 0, 30, True, 3],
            ['MTTo', "Other organs mean transit time", 20.0, "sec", 0, 60, True, 3],
            ['TTDo', "Other organs transit time dispersion", 10.0, "sec", 0, np.inf, True, 2],
            ['MTTe', "Storage compartment mean transit time", 120.0, "sec", 0, 800.0, True, 3],
            ['El', "Storage compartmentleakage  fraction", 0.15, "", 0, 0.50, True, 4],
            ['Ee',"Kidney & Liver Extraction fraction", 0.05,"", 0.01, 0.15, True, 4],
        ]      

    def predict_signal(self):
        R1 = self.predict_R1()
        p = self.p.value
        signal = dcmri.signalSPGRESS(p.TR, p.FA1, R1, p.S01)
        k2 = np.nonzero(self.t() >= self.t0[1]-self.t0[0])[0]
        signal[k2] = dcmri.signalSPGRESS(p.TR, p.FA2, R1[k2], p.S02)
        return signal
    
    def set_pars(self, TR, FA):
        self.p = self.pars_2scan(TR, FA) + self.pars()

    def export_pars(self, export_pars):
        p = self.p.value
        export_pars.drop(['TR','FA1','FA2','S01'], inplace=True)
        export_pars.loc['El', ['value', 'unit']] = [100*p.El, '%']
        export_pars.loc['Ee', ['value', 'unit']] = [100*p.Ee, '%']
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

