import os
import time
import pandas as pd
import numpy as np

import plot
from models.pbpk_aorta import OneScan as Model

structure = 'aorta'

def read(time, weight, params):
    print('Reading ' + structure + '...')
    result = Model(
        weight = weight,
        dose = 0.2, #mL/kg  
        conc = 0.5, # mmol/ml 
        rate = 1, # ml/sec
        TR = 4.80/1000.0, #sec
        FA = 17.0, #deg
        tdce = time,   
    )
    result.read_csv(params)
    return result


def fit(time, signal, weight):
    print('Fitting ' + structure + '...')
    result = Model(
        weight = weight,
        agent = 'Dotarem',
        dose = 0.2, #mL/kg  
        conc = 0.5, # mmol/ml 
        rate = 1, # ml/sec
        TR = 4.80/1000.0, #sec
        FA = 17.0, #deg
        tdce = time,   
        Sdce = signal,
        R1molli = 1000/1463.0,
        callback = False,
        ptol = 1e-3,
        tstop = None,
    )
    result.estimate_p()
    print(structure + ' goodness of fit: ', result.goodness())
    result.fit_p()
    print(structure + ' goodness of fit: ', result.goodness())
    return result


def fit_data(datadir, resultsdir):
    datafile = os.path.join(datadir, 'Overview.xlsx')
    parfile = os.path.join(datadir, 'ROI size.xlsx')
    data = pd.read_excel(datafile, sheet_name='Blad1')
    const = pd.read_excel(parfile, sheet_name='Blad1')
    weight = pd.read_excel(parfile, sheet_name='Blad2')
    output = pd.DataFrame(columns=['subject','visit','structure','name','value','unit'])
    for visit in ['1','2']:
        for s in ['1','2','3','4']:
            print('Fitting ', s, visit)
            curve = 'Dog' + s + '.' + visit + ' AIF'
            wght = weight[weight.Dog==int(s)].weight.values[0]
            time = data['Time'].values
            result = fit(time, data[curve].values, wght)
            result.to_csv(os.path.join(resultsdir, curve + '.csv'))
            result.plot_fit(save=True, show=False, path=resultsdir, prefix=curve)
            pars = result.export_p()
            pars['subject'] = s
            pars['visit'] = visit
            pars['structure'] = structure
            output = pd.concat([output, pars])
    try:
        output['parameter'] = output.index
        output.to_csv(os.path.join(resultsdir, 'parameters.csv'), index=False)
        output.to_pickle(os.path.join(resultsdir, 'parameters.pkl'))
    except:
        print("Can't write to parameter file ")
        print("Please close the file before saving data")
    return os.path.join(resultsdir, 'parameters.csv')


def main(datadir, resultsdir):

    start = time.time()

    fit_data(datadir, resultsdir)

    output_file = os.path.join(resultsdir, 'parameters.csv')
    plot.create_bar_chart(output_file, ylim={})

    print('Fit pbpk aorta 1-scan calculation time (mins): ', (time.time()-start)/60)
