import math
import time
import os
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

import dcmri

def rp(field_strength, agent='Primovist'): # Gadoxetate
# Szomolanyi P, Rohrer M, Frenzel T, Noebauer-Huhmann IM, Jost G, Endrikat J, Trattnig S, Pietsch H. Comparison of the Relaxivities of Macrocyclic Gadolinium-Based Contrast Agents in Human Plasma at 1.5, 3, and 7 T, and Blood at 3 T. Invest Radiol. 2019 Sep;54(9):559-564. doi: 10.1097/RLI.0000000000000577.
    field = math.floor(field_strength)
    if agent=='Primovist':
        if field == 1.5: return 8.1     # relaxivity of blood in Hz/mM
        if field == 3.0: return 6.4     # relaxivity of blood in Hz/mM
        if field == 4.0: return 6.4     # relaxivity of blood in Hz/mM
        if field == 7.0: return 6.2     # relaxivity of blood in Hz/mM
        if field == 9.0: return 6.1     # relaxivity of blood in Hz/mM 
    if agent=='Clariscan':
        if field == 3.0: return 2.72 
    if agent=='Dotarem':
        if field == 3.0: return 2.72 
    msg = 'No relaxivity data for ' + agent + ' at ' + str(field_strength) + ' T.'
    raise ValueError(msg)

def rh(field_strength):
    field = math.floor(field_strength)
    if field == 1.5: return 14.6    # relaxivity of hepatocytes in Hz/mM
    if field == 3.0: return 9.8     # relaxivity of hepatocytes in Hz/mM
    if field == 4.0: return 7.6     # relaxivity of hepatocytes in Hz/mM
    if field == 7.0: return 6.0     # relaxivity of hepatocytes in Hz/mM
    if field == 9.0: return 6.1     # relaxivity of hepatocytes in Hz/mM

def R1_blood(field_strength=3.0, Hct=0.45):
    field = math.floor(field_strength)
    if field == 1.5: return 1000.0 / 1480.0    # aorta R1 in 1/sec 
    if field == 3.0: return 0.52 * Hct + 0.38  # Lu MRM 2004 

def R1_liver(field_strength=3.0):
    field = math.floor(field_strength)
    if field == 1.5: return 1000.0/602.0     # liver R1 in 1/sec (Waterton 2021)
    if field == 3.0: return 1000.0/752.0     # liver R1 in 1/sec (Waterton 2021)
    if field == 4.0: return 1.281     # liver R1 in 1/sec (Changed from 1.285 on 06/08/2020)
    if field == 7.0: return 1.109     # liver R1 in 1/sec (Changed from 0.8350 on 06/08/2020)
    if field == 9.0: return 0.920     # per sec - liver R1 (https://doi.org/10.1007/s10334-021-00928-x)

def R1_kidney(field_strength=3.0):
    # Reference values average over cortext and medulla from Cox et al
    # https://academic.oup.com/ndt/article/33/suppl_2/ii41/5078406
    field = math.floor(field_strength)
    if field == 1.5: return 1000.0/((1024+1272)/2)
    if field == 3.0: return 1000.0/((1399+1685)/2)


class FitFunc():

    def save_path(self, path):
        if path is None:
            path = os.path.dirname(__file__)
            path = os.path.join(path, 'results')
            if not os.path.isdir(path):
                os.mkdir(path)
        if not os.path.exists(path):
            os.makedirs(path)
        return path
    
    def _set_df(self):
        cols = ['symbol', "name", "initial value", "unit", "lower bound", "upper bound", "fit", "digits"]
        self.p = pd.DataFrame(self.p, columns=cols)
        self.p.set_index('symbol', inplace=True)
        self.p['value'] = self.p['initial value']
        cols = cols[1:]
        cols.insert(2, 'value')
        self.p = self.p[cols]
    
    def export_p(self, path=None, prefix=None):
        export_pars = self.p.drop(['initial value','lower bound','upper bound','fit','digits'], axis=1)
        export_pars = self.export_pars(export_pars)
        if path is not None: 
            if not os.path.isdir(path):
                os.makedirs(path)
            pre = ''
            if prefix is not None:
                pre += prefix + '_'
            save_file = os.path.join(path, pre + self.__class__.__name__ + '_fitted_parameters.csv')
            try:
                export_pars.to_csv(save_file)
            except:
                print("Can't write to file ", save_file)
                print("Please close the file before saving data")
        return export_pars 
    

    def fit_p(self):
        self.it = 0
        start = time.time()
        try:
            _,y = self.xy_fitted() 
            self.p.loc[self.p.fit, 'value'], self.pcov = curve_fit( 
                self._fit_function, None, y, 
                p0 = self.p.loc[self.p.fit, 'value'].values, 
                #sigma = self.sigma[self.valid][self.xind],
                bounds = (
                    self.p.loc[self.p.fit, 'lower bound'].values,
                    self.p.loc[self.p.fit, 'upper bound'].values,
                    ),
                ftol=self.ftol,
                gtol=self.gtol,
                xtol=self.ptol,
            )
        except ValueError as e:
            print(e)
        except RuntimeError as e:
            print(e)
        self.predict_data()
        end = time.time()
        if self.callback:
            print('Finished fitting ' + self.__class__.__name__)
            print('>> Number of iterations: ' + str(self.it))
            print('>> Calculation time (mins): ' + str((end-start)/60))

    
    def plabel(self):
        label = ''
        for _, p in self.p.iterrows():
            # v = str(p.value).split('.')
            # digits = p.digits-len(v[0])
            # if digits >= 0:
            #     v = round(p.value, digits)
            # else:
            #     v = p.value
            v = round(p.value, p.digits)
            label += '\n'
            label += p.name + " = " + str(v) + " " + p.unit
        return label
    
    def to_csv(self, file):
        path = os.path.dirname(file)
        if not os.path.isdir(path):
            os.makedirs(path)
        try:
            self.p.to_csv(file)
        except:
            print("Can't write to file ", file)
            print("Please close the file before saving data.")   

    def read_csv(self, file):
        try:
            p = pd.read_csv(file, index_col='symbol')   
        except:
            msg = 'Cannot read model parameters from file ' + file
            msg += '\nPlease check if the file exists and is not open in another program.'
            raise RuntimeError(msg)
        if p.columns.to_list() != self.p.columns.to_list():
            msg = 'Parameters read from file have the incorrect column headers'
            raise ValueError(msg)
        if p.index.to_list() != self.p.index.to_list():
            msg = 'Parameters read from file have the incorrect row headers'
            raise ValueError(msg)
        self.p = p


class SuperModel(FitFunc):

    def __init__(self,
            # Constants needed to predict pseudocontinuous signal
            dt = 0.5,                   # Internal time resolution (sec)
            tmax = 40*60,               # Acquisition time (sec)
            field_strength = 3.0,       # Field strength (T)
            # Additional constants used to predict data
            tdce = None,
            tmolli = None,
            t0 = None,
            # Constants used in model fitting
            Sdce = None,
            R1molli = None,
            callback = False,
            ptol = 1e-8,
            dcevalid = None,
            tstart = None, 
            tstop = None, 
            cweight = None,
            ):
        self.dt = dt
        self.tmax = tmax 
        self.field_strength = field_strength
        # Constants needed for data prediction
        self.tdce = tdce
        self.tR1 = tmolli
        self.t0 = t0
        self.callback = callback
        # Constants needed for model fitting
        self.Sdce = Sdce
        self.vR1 = R1molli
        self.ftol = 1e-8
        self.ptol = ptol
        self.gtol = 1e-8
        self._set_xind(dcevalid, tstart, tstop)
        self._set_cweight(cweight)

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
            ['S01', "Signal amplitude S0", 1200, "a.u.", 0, np.inf, False, 4],
            ['S02', "Signal amplitude S0", 1200, "a.u.", 0, np.inf, True, 4],
            #['S02', "Signal amplitude S0", 1200, "a.u.", 0, np.inf, True, 4],
        ]
    
    def pars_1scan(self, TR, FA):
        return [
            ['TR', "Repetition time", TR, "sec", 0, np.inf, False, 4],
            ['FA', "Flip angle", FA, "deg", 0, 180, False, 4],
            ['S0', "Signal amplitude S0", 1200, "a.u.", 0, np.inf, False, 4],
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
        self.p.at['R10','value'] = self.vR1[0]
        p = self.p.value
        Sref = dcmri.signalSPGRESS(p.TR, p.FA, p.R10, 1)
        baseline = np.nonzero(self.tdce <= self.BAT)[0]
        n0 = baseline.size
        if n0 == 0: 
            n0 = 1
        S0 = np.mean(self.Sdce[:n0]) / Sref
        self.p.at['S0','value'] = S0
        
    def estimate_p_tissue_2scan(self):
        self.p.at['R10','value'] = self.vR1[0]
        p = self.p.value

        k1 = self.tdce < self.t0[1]-self.t0[0]
        tdce1 = self.tdce[k1]
        Sdce1 = self.Sdce[k1]
        BAT = self.BAT[0]
        Sref = dcmri.signalSPGRESS(p.TR, p.FA1, p.R10, 1)
        baseline = np.nonzero(tdce1 <= BAT)[0]
        n0 = baseline.size
        if n0 == 0: 
            n0 = 1
        S01 = np.mean(Sdce1[:n0]) / Sref
        self.p.at['S01','value'] = S01

        k2 = self.tdce >= self.t0[1]-self.t0[0]
        tdce2 = self.tdce[k2]
        Sdce2 = self.Sdce[k2]
        Sref = dcmri.signalSPGRESS(p.TR, p.FA2, self.vR1[2], 1)
        n0 = math.floor(60/(tdce2[1]-tdce2[0])) # 1 minute baseline
        S02 = np.mean(Sdce2[:n0]) / Sref
        self.p.at['S02','value'] = S02

    def plot_with_conc(self, xrange=None, legend=True, win='all', show=True, save=False, path=None, prefix=''):
        t = self.t()
        if xrange is None:
            xrange = [t[0],t[-1]]
        fig, (ax1, ax2) = plt.subplots(1,2,figsize=(10,4))
        fig.subplots_adjust(
                    #left=0.1,
                    #right=0.9,
                    #bottom=0.1,
                    #top = 0.9, 
                    wspace=0.3,
                    #hspace=1,
                    )
        #fig.suptitle("model fit " + win)
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
        #ax1.set_title('Signal')
        ax1.set(xlabel='Time (sec)', ylabel='MR Signal (a.u.)', xlim=xlim)
        ax1.plot(xi+tacq/2, yi, marker='o', color='gray', label='ignored data', linestyle = 'None')
        ax1.plot(xf+tacq/2, yf, 'ro', label='fitted data')
        ax1.plot(self.t(), self.predict_signal(), 'b-', label='fit' )
        ax1.plot([self.tdce[0]]+self.tR1[1:], self.s_molli(), 'gx', label='test data (MOLLI)')
        if legend:
            ax1.legend()

    def s_molli(self):
        p = self.p.value
        if len(self.vR1) == 2:
            return dcmri.signalSPGRESS(p.TR, p.FA, self.vR1, p.S0)
        else:
            return [
                dcmri.signalSPGRESS(p.TR, p.FA1, self.vR1[0], p.S01),
                dcmri.signalSPGRESS(p.TR, p.FA1, self.vR1[1], p.S01),
                dcmri.signalSPGRESS(p.TR, p.FA2, self.vR1[2], p.S02),
            ]

    def xy_fitted(self):
        x = np.concatenate([self.tdce, [self.tdce[0]]+self.tR1[1:]])
        y = np.concatenate([self.Sdce, self.s_molli()])
        x = x[self.valid][self.xind]
        y = y[self.valid][self.xind]
        return x, y 
    
    def xy_ignored(self):
        x = np.concatenate([self.tdce, [self.tdce[0]]+self.tR1[1:]])
        y = np.concatenate([self.Sdce, self.s_molli()])
        valid = np.full(x.shape, False)
        valid[self.valid[0][self.xind]] = True
        ind = np.where(valid==False)
        return x[ind], y[ind]

    def predict_data(self):
        signal = self.predict_signal()
        x = np.concatenate([self.tdce, [self.tdce[0]]+self.tR1[1:]])
        tacq = self.tdce[1]-self.tdce[0]
        self.yp = dcmri.sample(self.t(), signal, x, tacq)
        return self.yp

    def _set_cweight(self, wcal):
        if wcal == None:
            self.sigma = np.ones(len(self.tdce)+len(self.tR1))
        else:
            n = len(self.tdce) + len(self.tR1)
            ncal = len(self.vR1)
            ndat = n - ncal
            weight_per_cal_pt = wcal/ncal
            weight_per_dat_pt = (1-wcal)/ndat
            weights = np.full(n, weight_per_dat_pt)
            weights[-ncal:] = weight_per_cal_pt
            if 0 in weights:
                msg = 'Weights cannot be zero anywhere. '
                msg += '\nUse set_valid() or set_xrange() to exclude data points from the fit.'
                raise ValueError(msg)
            self.sigma = np.divide(1, weights)

    def _set_xind(self, dcevalid, tstart, tstop):
        # Indices of valid time points
        t = np.concatenate([self.tdce, [self.tdce[0]]+self.tR1[1:]])
        if dcevalid is None:
            valid = np.full(t.shape, True)
        else:
            valid = np.concatenate([dcevalid, np.full(len(self.tR1), True)])
        if tstart is None:
            tstart = np.amin(t)
        if tstop is None:
            tstop = np.amax(t)
        self.valid = np.where(valid)
        t = t[self.valid]
        self.xind = np.nonzero((t>=tstart) & (t<=tstop))[0]

    def t(self): # internal time
        return np.arange(0, self.tmax+self.dt, self.dt)

