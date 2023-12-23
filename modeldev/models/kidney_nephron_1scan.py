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
        ca = self.cb/(1-self.Hct)
        # Derived constants from (vp, Tp, f, r)
        # Fp + rFt = Ft + Fv => Fv = Fp-(1-r)Ft
        # Fpca = [Fp+rFt]cp = (1+rf)Fpcp => cp = ca/(1+rf)
        # Tp = vp/(Fp + rFt) = vp/Fp/(1+rf) => Fp = (1+rf)vp/Tp
        # Ft = fFp
        vp = (1-self.Hct)*p.vb
        Fp = (1+p.r*p.FF)*vp/p.Tp 
        Ft = p.FF*Fp
        vt = 1-p.vb
        Fu = (1-p.r)*Ft
        # Tissue concentration in the plasma
        cp = dcmri.prop_comp(t, ca, p.Tp)/(1+p.r*p.FF)
        self.Cp = vp*cp
        # Tissue concentration in the tubuli
        self.Ct = dcmri.conc_nephc(t, Ft*cp, p.r, Ft/vt, p.dr/vt, n=50)
        # Return R
        rp = lib.rp(self.field_strength)
        R1 = p.R10 + rp*(self.Cp + self.Ct)
        # Calculate signal
        signal = dcmri.signalSPGRESS(p.TR, p.FA, R1, p.S0)
        return signal
    
    def set_pars(self, TR, FA):
        self.p = self.pars_1scan(TR, FA) + [
            ['R10', "Baseline R1", lib.R1_kidney(), "1/sec", 0, np.inf, False, 4],
            # Inlets
            ['Tdel', "Arterial delay time", 2.0, 'sec', 0, 5.0, False, 6],  
            # Kidney tissue
            ['vb', "Blood volume", 0.3, 'mL/mL', 0.01, 0.6, True, 6],
            ['Tp', "Plasma transit time", 5, 'sec', 0, 8, True, 6],
            ['FF', "Filtration Fraction", 0.1, '', 0.01, 0.5, True, 6],
            ['r', "Reabsorption fraction", 0.95, '', 0.5, 1.0, True, 6],
            ['dr', "Diffusion rate", 1e-4, '1/sec', 0, 1, True, 6],
        ]

    def export_pars(self, export_pars):
        p = self.p.value
        export_pars.drop(['TR','FA','S0'], axis=0, inplace=True)
        # Convert to conventional units
        export_pars.loc['vb', ['value', 'unit']] = [100*p.vb, 'mL/100mL']
        export_pars.loc['FF', ['value', 'unit']] = [100*p.FF, '%']
        export_pars.loc['r', ['value', 'unit']] = [100*p.r, '%']
        vp = (1-self.Hct)*p.vb
        Fp = (1+p.r*p.FF)*vp/p.Tp 
        Ft = p.FF*Fp
        vt = 1-p.vb
        Fu = (1-p.r)*Ft
        Fb = Fp/(1-self.aorta.Hct)
        export_pars.loc['Fb'] = ["Renal blood flow", 6000*Fb, 'mL/min/100mL']
        export_pars.loc['Ft'] = ["Tubular flow", 6000*Ft, 'mL/min/100mL']
        export_pars.loc['Fu'] = ["Urine flow", 6000*Fu, 'mL/min/100mL']    
        export_pars.loc['vt'] = ["Tubular volume fraction", 100*vt, 'mL/100mL']
        export_pars.loc['Tt'] = ["Tubular transit time", vt/Fu/60, 'min']
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

