import os
import time
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

class CurveFit():

    # optimization settings
    it = 0
    callback = False
    ftol = 1e-8
    ptol = 1e-8
    gtol = 1e-8

    # defaults
    x = np.arange(0, 1, 0.05)
    yp = None
    valid = np.where(np.full(x.shape, True))
    xname = 'x'
    xunit = 'unit'
    yname = 'y'
    yunit = 'unit'
    xind = None
    xi = None
    xf = None

    # Define constants (if any)
    power = 3.0

    def function(self, x, p):

        return p.a*x**self.power + p.b      

    def parameters(self):
        
        return [
            ['a', "slope", 1, "au", -np.inf, np.inf, True, 3],
            ['b', "intercept", 0, 'au', -np.inf, np.inf, True, 3],
        ]

    def __init__(self):

        self.reset_p()

    def reset_p(self):
        """Reset parameters to factory settings"""

        p = self.parameters()
        if type(p[0]) is str:
            p = [[variable, variable, 0, 'not provided', -np.inf, np.inf, True, 3] for variable in p]
        cols = ['symbol', "name", "initial value", "unit", "lower bound", "upper bound", "fit", "digits"]
        self.p = pd.DataFrame(p, columns=cols)
        self.p.set_index('symbol', inplace=True)
        self.p['value'] = self.p['initial value']
        cols = cols[1:]
        cols.insert(2, 'value')
        self.p = self.p[cols]

    def predict_y(self, x=None):

        if x is not None:
            self.set_x(x)
        self.yp = self.function(self.x, self.p.value)
        return self.yp

    def estimate_p(self):
        pass

    def fit(self, parameter, value=True):
        self.p.at[parameter, 'fit'] = value

    def plabel(self):
        label = ''
        for _, p in self.p.iterrows():
            v = str(p.value).split('.')
            digits = p.digits-len(v[0])
            if digits >= 0:
                v = round(p.value, digits)
            else:
                v = p.value
            label += '\n'
            label += p.name + " = " + str(v) + " " + p.unit
        return label
    
    def set_p(self, p):
        self.p = p

    def set_x(self, x:np.ndarray, name=None, unit=None, valid=None):
        self.x = x 
        if name is not None:
            self.xname = name
        if unit is not None:
            self.xunit = unit
        self.set_valid(valid)
        
    def set_y(self, value=None, name=None, unit=None):
        if value is not None:
            self.y = value # must be a numpy array
            self.sigma = np.ones(len(value))
        if name is not None:
            self.yname = name
        if unit is not None:
            self.yunit = unit

    def set_weights(self, weights):
        if 0 in weights:
            msg = 'Weights cannot be zero anywhere. '
            msg += '\nUse set_valid() or set_xrange() to exclude data points from the fit.'
            raise ValueError(msg)
        self.sigma = np.divide(1, weights)

    def set_xy(self, x, y):
        self.set_x(x)
        self.set_y(y)

    @property
    def xlabel(self):
        return self.xname + ' (' + self.xunit + ')'

    @property
    def ylabel(self):
        return self.yname + ' (' + self.yunit + ')'

    @property
    def parameter_values(self):
        return self.p['value'].values
    
    def set_valid(self, valid=None): 
        # By default all points are valid (used in the fit)
        if valid is None:
            valid = np.full(self.x.shape, True)
        self.valid = np.where(valid)
        self.set_xrange()

    def set_xrange(self, xi=None, xf=None):
        # By default the whole range is fitted
        if xi is None:
            if self.xi is None:
                self.xi = np.amin(self.x)
        else:
            self.xi = xi
        if xf is None:
            if self.xf is None:
                self.xf = np.amax(self.x)
        else:
            self.xf = xf
        x = self.x[self.valid]
        self.xind = np.nonzero((x>=self.xi) & (x<=self.xf))[0]

    def xy_fitted(self):
        x = self.x[self.valid][self.xind]
        y = self.y[self.valid][self.xind]
        return x, y

    def xy_ignored(self):
        valid = np.full(self.x.shape, False)
        valid[self.valid[0][self.xind]] = True
        ind = np.where(valid==False)
        return self.x[ind], self.y[ind]

    def _fit_function(self, _, *params):

        self.it += 1 
        if self.callback:
            p = ' '.join([str(p) for p in params])
            print('Fitting ' + self.__class__.__name__ + ', iteration: ' + str(self.it))
            print('>> Parameter values: ' + p)

        self.p.loc[self.p.fit,'value'] = params
        self.predict_y() # Create a prediction everywhere
        return self.yp[self.valid][self.xind]

    def goodness(self):
        if self.yp is None:
            self.predict_y()
        yref = self.y[self.valid][self.xind]
        ydist = np.linalg.norm(self.yp[self.valid][self.xind] - yref)
        return 100*ydist/np.linalg.norm(yref)

    def fit_p(self, x=None, y=None):

        self.it = 0
        start = time.time()

        if x is not None:
            self.set_x(x)
        if y is not None:
            self.set_y(y)
        try:
            x, y = self.xy_fitted() # note x is ignored here - not used by _fit_function
            self.p.loc[self.p.fit, 'value'], self.pcov = curve_fit( 
                self._fit_function, x, y, 
                p0 = self.p.loc[self.p.fit, 'value'].values, 
                sigma = self.sigma[self.valid][self.xind],
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
        self.predict_y()
        end = time.time()

        if self.callback:
            print('Finished fitting ' + self.__class__.__name__)
            print('>> Number of iterations: ' + str(self.it))
            print('>> Calculation time (mins): ' + str((end-start)/60))

    def plot_prediction(self, show=True, save=False, path=None):
        name = self.__class__.__name__
        plt.title(name + ' - model prediction')
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        plt.plot(self.x, self.yp, 'g-', label='prediction ' + self.plabel())
        plt.legend()
        if save:
            if path is None:
                path = self.path()
            plt.savefig(fname=os.path.join(path, name + '_prediction' + '.png'))
        if show:
            plt.show()
        else:
            plt.close()

    def plot_data(self, show=True, save=False, path=None): 

        name = self.__class__.__name__
        plt.title(name + " - data")
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        plt.plot(self.x, self.y, 'ro', label='data')
        plt.legend()
        if save:
            if path is None:
                path = self.path()            
            plt.savefig(fname=os.path.join(path, name + '_data' + '.png'))
        if show:
            plt.show()
        else:
            plt.close()

    def plot_fit(self, xrange=None, show=True, save=False, path=None, prefix=''): 
        if self.yp is None:
            self.predict_y()
        if xrange is None:
            x0 = self.x[0]
            x1 = self.x[-1]
            win_str = ''
        else:
            x0 = xrange[0]
            x1 = xrange[1]
            win_str = ' [' + str(round(x0)) + ', ' + str(round(x1)) + ']'
        xf, yf = self.xy_fitted()
        xi, yi = self.xy_ignored()
        name = self.__class__.__name__
        plt.title(name + " - model fit"+ win_str)
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        plt.xlim([x0, x1])
        plt.plot(xf, yf, 'ro', label='data (fitted)', linestyle='None')
        plt.plot(xi, yi, marker='o', color='gray', label='data (ignored)', linestyle='None')
        plt.plot(self.x, self.yp, 'b--', label='fit ' + self.plabel())
        plt.legend()
        if save:
            if path is None:
                path = self.path()      
            plt.savefig(fname=os.path.join(path, prefix+name + '_fit' + win_str + '.png'))
        if show:
            plt.show()
        else:
            plt.close()

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
            print('Cannot read model parameters from file ', file)
            print('Please check if the file is open in another program.')
        if p.columns.to_list() != self.p.columns.to_list():
            msg = 'Parameters read from file have the incorrect column headers'
            raise ValueError(msg)
        if p.index.to_list() != self.p.index.to_list():
            msg = 'Parameters read from file have the incorrect row headers'
            raise ValueError(msg)
        self.p = p

    def export_p(self, path=None, prefix=None):
        self.set_export_pars()
        if path is not None: 
            if not os.path.isdir(path):
                os.makedirs(path)
            pre = ''
            if prefix is not None:
                pre += prefix + '_'
            save_file = os.path.join(path, pre + self.__class__.__name__ + '_fitted_parameters.csv')
            try:
                self.export_pars.to_csv(save_file)
            except:
                print("Can't write to file ", save_file)
                print("Please close the file before saving data")
        return self.export_pars

    def set_export_pars(self):
        self.export_pars = self.p.copy()

    def path(self):
        path = os.path.dirname(__file__)
        path = os.path.join(path, 'results')
        if not os.path.isdir(path):
            os.mkdir(path)
        return path



    # Export curves
    # -------------
    # df_results = pd.DataFrame({"Time fit (s)": time})
    # df_results["Liver fit (a.u.)"] = subject.liver_signal
    # df_output = pd.concat([df_data, df_results], axis=1)
    # save_file = data.results_path() + 'fit_' + filename + ".csv"
    # try:
    #     df_output.to_csv(save_file)
    # except:
    #     print("Can't write to file ", save_file)
    #     print("Please close the file before saving data")


class BiExponential(CurveFit):

    def function(self, x, p):
        return p.A*np.exp(-p.a*x) + p.B*np.exp(-p.b*x)

    def parameters(self):
        return ['A', 'a', 'B', 'b']


def test_biexp_fit():

    x = np.arange(0, 1, 0.05)
    y = 3*x**2 + 200

    c = BiExponential()
    c.p['upper bound'] = [np.inf,1,100,1]
    c.fit_p(x,y)
    c.plot_fit(save=True)

def test_curve_fit():

    x = np.arange(0, 1, 0.05)
    y = 3*x**2 - 200

    c = CurveFit()
    c.set_x(x)
    c.predict_y()
    c.plot_prediction(save=True)
    c.set_y(y)
    c.plot_data(save=True)
    c.fit_p()
    c.plot_fit(save=True)
    c.export_p()

def test_range():
    x = np.arange(0, 1, 0.05)
    y = 3*x**2 + 200
    c = BiExponential()
    c.callback=True
    c.ptol=1e-1
    c.set_x(x)
    c.set_y(y)
    c.set_xrange(0.6,0.8)
    c.fit_p()
    c.plot_fit(save=False)

def test_read_write():
    file = os.path.dirname(__file__)
    file = os.path.join(file, 'tmp', 'biexp.csv')
    x = np.arange(0, 1, 0.05)
    y = 3*x**2 + 200
    c = BiExponential()
    c.ptol=1e-1
    c.set_x(x)
    c.set_y(y)
    c.fit_p()
    v0 = c.p.value.to_list()
    c.to_csv(file)
    c.read_csv(file)
    v1 = c.p.value.to_list()
    assert np.array_equal(np.round(v1,1), np.round(v0,1))

if __name__ == "__main__":
    #test_biexp_fit()
    #test_curve_fit()
    #test_range()
    test_read_write()
