import numpy as np
import dcmri
import models.lib as lib

class Kidney(lib.SuperModel):
    def __init__(self, 
            # Constants needed to predict pseudocontinuous signal
            cb = None,
            Hct = 0.45,
            TR = 3.71/1000.0,           # Repetition time (sec)
            FA = 15.0,                  # Nominal flip angle (degrees)
            # Constants used in model fitting
            BAT = None,
            **kwargs,
        ):
        super().__init__(**kwargs)
        # Essential constants
        self.cb = cb
        self.Hct = Hct
        # Constants needed for model fitting
        self.BAT = BAT
        self.set_pars(TR, FA)
        self._set_df()

    def predict_R1(self):
        p = self.p.value
        t = self.t()
        # Propagate through arterial tree
        #ca = dcmri.propagate_delay(t, self.aorta.cb, p.Tdel)
        ca = self.cb/(1-self.Hct)
        # Tissue concentration in the plasma
        # Fp, Tp, Ft, Tt
        self.Cp = dcmri.res_comp(t, p.Fp*ca, p.Tp)
        vp = p.Tp*(p.Fp+p.Ft)
        cp = self.Cp/vp
        # Tissue concentration in the tubuli
        #self.Ct = dcmri.res_plug(t, p.Ft*cp, p.Tt)
        self.Ct = dcmri.res_comp(t, p.Ft*cp, p.Tt)
        # Return R
        rp = lib.rp(self.field_strength)
        R1 = p.R10 + rp*(self.Cp + self.Ct)
        return R1 
    
    def pars(self):
        return [
            # Signal parameters
            ['R10', "Baseline R1", lib.R1_kidney(), "1/sec", 0, np.inf, False, 4],
            # Inlets
            #['Tdel', "Arterial delay time", 2.0, 'sec', 0, 5.0, False, 6],  
            # Tissue
            ['Fp', "Plasma flow", 200/6000, 'mL/sec/mL', 0, np.inf, True, 6],
            ['Tp', "Plasma mean transit time", 5, 'sec', 0.0, 8, True, 6],
            ['Ft', "Tubular flow", 30/6000, 'mL/sec/mL', 0, np.inf, True, 6],
            ['Tt', "Tubular mean transit time", 120, 'sec', 1, np.inf, True, 6],
        ]
    
    def export_pars(self, export_pars):
        p = self.p.value
        # Convert to conventional units
        export_pars.loc['Fp', ['value', 'unit']] = [6000*p.Fp, 'mL/min/100mL']
        export_pars.loc['Ft', ['value', 'unit']] = [6000*p.Ft, 'mL/min/100mL']
        export_pars.loc['Tt', ['value', 'unit']] = [p.Tt/60, 'min']
        # Add derived parameters 
        export_pars.loc['vp'] = ["Plasma volume fraction", 100*p.Tp*(p.Fp+p.Ft), 'mL/100mL']         
        export_pars.loc['FF'] = ["Filtration Fraction", 100*p.Ft/p.Fp, '%']
        return export_pars

    def plot_conc_fit(self, ax2, xlim, legend):
        t = self.t()
        ax2.set_title('Reconstructed concentration')
        ax2.set(xlabel='Time (sec)', ylabel='Tissue concentration (mM)', xlim=xlim)
        ax2.plot(t, 0*t, color='gray')
        ax2.plot(t, self.Cp, 'g-', label='Plasma')
        ax2.plot(t, self.Ct, 'g--', label='Tubuli')
        ax2.plot(t, self.Cp+self.Ct, 'b-', label=self.plabel())
        if legend:
            ax2.legend() 


class KidneyOneScan(Kidney):

    def predict_signal(self):
        R1 = self.predict_R1()
        p = self.p.value
        signal = dcmri.signalSPGRESS(p.TR, p.FA, R1, p.S0)
        return signal
    
    def set_pars(self, TR, FA):
        self.p = self.pars_1scan(TR, FA) + super().pars()

    def export_pars(self, export_pars):
        export_pars.drop(['TR','FA','S0'], axis=0, inplace=True)
        return super().export_pars(export_pars)
    
    def estimate_p(self):
        self.estimate_p_tissue_1scan()  

    def plot_fit(self, **kwargs):
        self.plot_fit_tissue_1scan(**kwargs)


class KidneyTwoScan(Kidney):

    def predict_signal(self):
        R1 = self.predict_R1()
        p = self.p.value
        signal = dcmri.signalSPGRESS(p.TR, p.FA1, R1, p.S01)
        k2 = np.nonzero(self.t() >= self.t0[1]-self.t0[0])[0]
        signal[k2] = dcmri.signalSPGRESS(p.TR, p.FA2, R1[k2], p.S02)
        return signal
    
    def set_pars(self, TR, FA):
        self.p = self.pars_1scan(TR, FA) + super().pars()

    def export_pars(self, export_pars):
        export_pars.drop(['TR','FA1','FA2','S01'],inplace=True)
        return super().export_pars(export_pars)

    def estimate_p(self):
        self.estimate_p_tissue_2scan()
        
    def plot_fit(self, **kwargs):
        self.plot_fit_tissue_2scan(**kwargs)





