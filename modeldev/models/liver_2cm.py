import numpy as np
import dcmri
import models.lib as lib

class Liver(lib.SuperModel):
    def __init__(self, 
            # Constants needed to predict pseudocontinuous signal
            cb = None,
            Hct = 0.45,
            TR = 3.71/1000.0,           # Repetition time (sec)
            FA = 15.0,                  # Nominal flip angle (degrees)
            # Constants used in model fitting
            BAT = None,
            liver_volume = 1000, # mL
            **kwargs,   
        ):
        super().__init__(**kwargs)
        # Essential constants
        self.cb = cb
        self.Hct = Hct
        # Constants needed for model fitting
        self.BAT = BAT
        self.liver_volume = liver_volume
        self.set_pars(TR,FA)
        self._set_df()

    def predict_R1(self):
        p = self.p.value
        t = self.t()
        # Propagate through the gut
        ca = dcmri.propagate_dd(t, self.cb, p.Tdel, p.Te)
        # Tissue concentration in the extracellular space
        self.Ce = p.ve*ca/(1-self.Hct)
        # Tissue concentration in the hepatocytes
        self.Ch = dcmri.res_comp(t, p.k_he*ca, 1/p.Kbh)
        # Return R
        rp = lib.rp(self.field_strength)
        rh = lib.rh(self.field_strength)
        R1 = p.R10 + rp*self.Ce + rh*self.Ch
        return R1
    
    def pars(self):
        return [
            # Signal parameters
            ['R10', "Baseline R1", lib.R1_liver(), "1/sec", 0, np.inf, False, 4],
            # Inlets
            ['Tdel', "Gut delay time", 5.0, 'sec', 0, 20.0, True, 6],  
            ['Te', "Extracellular transit time", 30.0, 'sec', 0, 60, True, 6],
            # Liver tissue
            ['ve', "Extracellular volume fraction", 0.3, 'mL/mL', 0.01, 0.6, True, 6],
            ['k_he', "Hepatocellular uptake rate", 20/6000, 'mL/sec/mL', 0, np.inf, True, 6],
            ['Kbh', "Biliary tissue excretion rate", 1/(30*60), 'mL/sec/mL', 1/(10*60*60), 1/(10*60), True, 6],
        ]
    
    def export_pars(self, export_pars):
        p = self.p.value
        # Convert to conventional units
        export_pars.loc['Te', ['value', 'unit']] = [p.Te/60, 'min']
        export_pars.loc['ve', ['value', 'unit']] = [100*p.ve, 'mL/100mL']
        export_pars.loc['k_he', ['value', 'unit']] = [6000*p.k_he, 'mL/min/100mL']
        export_pars.loc['Kbh', ['value', 'unit']] = [6000*p.Kbh, 'mL/min/100mL']
        # Add derived parameters 
        export_pars.loc['Khe'] = ["Hepatocellular tissue uptake rate", 6000*p.k_he/p.ve, 'mL/min/100mL']         
        export_pars.loc['k_bh'] = ["Biliary excretion rate", 6000*p.Kbh*(1-p.ve), 'mL/min/100mL']
        export_pars.loc['Th'] = ["Hepatocellular transit time", np.divide(1, p.Kbh)/60, 'min']
        export_pars.loc['CL_l'] = ['Liver blood clearance', 60*p.k_he*self.liver_volume, 'mL/min']
        return export_pars
    
    def plot_data_fit(self, ax1, xlim, legend):
        xf, yf = self.xy_fitted()
        xi, yi = self.xy_ignored()
        tacq = self.tdce[1]-self.tdce[0]
        #ax1.set_title('Signal')
        ax1.set(xlabel='Time (min)', ylabel='MR Signal (a.u.)', xlim=np.array(xlim)/60)
        ax1.plot((xi+tacq/2)/60, yi, marker='o', color='gray', label='ignored data', linestyle = 'None')
        ax1.plot((xf+tacq/2)/60, yf, marker='o', color='cornflowerblue', label='fitted data', linestyle = 'None')
        ax1.plot(self.t()/60, self.predict_signal(), linestyle='-', color='darkblue', linewidth=3.0, label='fit' )
        ax1.plot(np.array([self.tdce[0]]+self.tR1[1:])/60, self.s_molli(), 'gx', label='MOLLI')
        if legend:
            ax1.legend()
    
    def plot_conc_fit(self, ax2, xlim, legend):
        t = self.t()/60
        ax2.set(xlabel='Time (min)', ylabel='Tissue concentration (mM)', xlim=np.array(xlim)/60)
        ax2.plot(t, 0*t, color='gray')
        ax2.plot(t, self.Ce, linestyle='-.', color='darkblue', label='Extracellular')
        ax2.plot(t, self.Ch, linestyle='--', color='darkblue', label='Hepatocyte')
        ax2.plot(t, self.Ce+self.Ch, linestyle='-', color='darkblue', linewidth=3.0, label='Tissue')
        if legend:
            ax2.legend()


class LiverOneScan(Liver):

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


class LiverTwoScan(Liver):            

    def predict_signal(self):
        R1 = self.predict_R1()
        p = self.p.value
        signal = dcmri.signalSPGRESS(p.TR, p.FA1, R1, p.S01)
        k2 = np.nonzero(self.t() >= self.t0[1]-self.t0[0])[0]
        signal[k2] = dcmri.signalSPGRESS(p.TR, p.FA2, R1[k2], p.S02)
        return signal
    
    def set_pars(self, TR, FA):
        self.p = self.pars_2scan(TR, FA) + super().pars()

    def export_pars(self, export_pars):
        export_pars.drop(['TR','FA1','FA2','S01'],inplace=True)
        return super().export_pars(export_pars)

    def estimate_p(self):
        self.estimate_p_tissue_2scan()

    def plot_fit(self, **kwargs):
        self.plot_fit_tissue_2scan(**kwargs)


    