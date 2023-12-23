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
            BAT = 0,
            **kwargs,
        ):
        super().__init__(**kwargs)
        # Essential constants
        self.cb = cb
        self.Hct = Hct
        # Constants needed for model fitting
        self.BAT = BAT
        # Variables
        self.set_pars(TR, FA)
        self._set_df()

    def predict_signal(self):
        p = self.p.value
        t = self.t()
        # Propagate through arterial tree
        #ca = dcmri.propagate_delay(t, self.aorta.cb, p.Tdel)
        ca = self.cb/(1-self.Hct)
        # Derived constants from (vb, Tp, ff, r)
        # Fp + rFt = Ft + Fv => Fv = Fp-(1-r)Ft
        # Fpca = [Fp+rFt]cp = (1+rf)Fpcp => cp = ca/(1+rf)
        # Tp = vp/(Fp + rFt) = vp/(Fp*(1+rf)) => Fp = vp/Tp/(1+rf)
        # Ft = fFp
        # f = E/(1-E) => f-Ef = E => E = 
        vp = (1-self.Hct)*p.vb
        Fp = vp/p.Tp/(1+p.r*p.FF)
        Ft = p.FF*Fp
        vt = 1-p.vb
        Fu = (1-p.r)*Ft
        Tt = vt/Fu
        # Tissue concentration in the plasma
        self.Cp = dcmri.res_comp(t, Fp*ca, p.Tp)
        cp = self.Cp/vp
        # Tissue concentration in the tubuli
        #self.Ct = dcmri.res_plug(t, p.Ft*cp, p.Tt)
        self.Ct = dcmri.res_comp(t, Ft*cp, Tt)
        # Return R
        rp = lib.rp(self.field_strength)
        R1 = p.R10 + rp*(self.Cp + self.Ct)
        # Calculate signal
        signal = dcmri.signalSPGRESS(p.TR, p.FA, R1, p.S0)
        return signal
    
    def set_pars(self, TR, FA):
        self.p = self.pars_1scan(TR, FA) + [
            ['R10', "Baseline R1", lib.R1_kidney(), "1/sec", 0, np.inf, False, 4],
            # Kidney tissue
            ['vb', "Blood volume", 0.3, 'mL/mL', 0.01, 0.99, True, 6],
            ['Tp', "Plasma transit time", 5, 'sec', 0, 15, True, 6],
            ['FF', "Filtration Fraction", 0.1, '', 0.01, 0.99, True, 6],
            ['r', "Reabsorption fraction", 0.95, '', 0.01, 0.99, True, 6],
        ]

    def export_pars(self, export_pars):
        p = self.p.value
        export_pars.drop(['TR','FA','S0'], axis=0, inplace=True)
        # Change units
        export_pars.loc['vb', ['value', 'unit']] = [100*p.vb, 'mL/100mL']
        export_pars.loc['FF', ['value', 'unit']] = [100*p.FF, '%']
        export_pars.loc['r', ['value', 'unit']] = [100*p.r, '%']
        # Add derived pars
        vp = (1-self.Hct)*p.vb
        Fp = vp/p.Tp/(1+p.r*p.FF)
        Ft = p.FF*Fp
        vt = 1-p.vb
        Fu = (1-p.r)*Ft
        Tt = vt/Fu
        Fb = Fp/(1-self.Hct)
        export_pars.loc['Fb'] = ["Renal blood flow", 6000*Fb, 'mL/min/100mL']
        export_pars.loc['Ft'] = ["Tubular flow", 6000*Ft, 'mL/min/100mL']
        export_pars.loc['Fu'] = ["Urine flow", 6000*Fu, 'mL/min/100mL']    
        export_pars.loc['vt'] = ["Tubular volume fraction", 100*vt, 'mL/100mL']
        export_pars.loc['Tt'] = ["Tubular transit time", Tt/60, 'min']
        return export_pars

    def estimate_p(self):
        self.estimate_p_tissue_1scan()  

    def plot_fit(self, **kwargs):
        self.plot_fit_tissue_1scan(**kwargs)

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


