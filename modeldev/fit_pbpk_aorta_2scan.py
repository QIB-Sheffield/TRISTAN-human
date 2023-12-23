import os
import time
import pandas as pd
import numpy as np

import data
import plot
from models.pbpk_aorta import TwoScan as Model

structure = 'aorta'
folder = 'pbpk_aorta_2scan'

def read(data, params):
    print('Reading ' + structure + '...')
    result = Model(
        weight = data['weight'],
        dose = [data['dose1'], data['dose2']], 
        tdce = np.concatenate([data['time1'], data['time2']]),  
        t0 = [data['t0'], data['t0']+data['time2'][0]],
    )
    result.read_csv(params)
    return result


def fit(data):
    print('Fitting ' + structure + '...')
    result = Model(
        weight = data['weight'],
        dose = [data['dose1'], data['dose2']],   
        tdce = np.concatenate([data['time1'], data['time2']]),  
        t0 = [data['t0'], data['t0']+data['time2'][0]],
        Sdce = np.concatenate([data['aorta1'], data['aorta2']]),
        R1molli = np.array([1000.0/data['T1aorta1'], 1000.0/data['T1aorta3']]),
        callback = False,
        ptol = 1e-3,
        dcevalid = np.concatenate([data['aorta_valid1'], data['aorta_valid2']]),
        tstop = None,
    )
    result.estimate_p()
    print(structure + ' goodness of fit: ', result.goodness())
    result.fit_p()
    print(structure + ' goodness of fit: ', result.goodness())
    return result



def fit_data(datapath, output_file):
    resultsfolder = os.path.dirname(output_file)
    output = pd.DataFrame(columns=['subject','visit','structure','name','value','unit'])
    for visit in ['baseline', 'rifampicin']:
        visitdatapath = os.path.join(datapath, visit)
        for s in os.listdir(visitdatapath):
            subj = os.path.join(visitdatapath, s)
            print('Fitting ', subj)
            subj_data = data.read(subj)
            result = fit(subj_data)
            result.to_csv(os.path.join(resultsfolder, s[:3] + '_' + visit + '.csv'))
            result.plot_fit(save=True, show=False, path=resultsfolder, prefix=s[:3] + '_' + visit)
            pars = result.export_p()
            pars['subject'] = s[:3]
            pars['visit'] = visit
            pars['structure'] = structure
            output = pd.concat([output, pars])
    try:
        output['parameter'] = output.index
        output.to_csv(output_file, index=False)
    except:
        print("Can't write to file ", output_file)
        print("Please close the file before saving data")

    return output_file


def main():

    start = time.time()

    #filepath = os.path.abspath("")
    filepath = os.path.dirname(__file__)
    datapath = os.path.join(filepath, 'data')
    results = os.path.join(filepath, 'results')
    resultspath = os.path.join(results, folder)
    output_file = os.path.join(resultspath, 'parameters.csv')

    fit_data(datapath, output_file)

    ylim = {}
    plot.create_bar_chart(output_file, ylim=ylim)
    plot.create_box_plot(output_file, ylim=ylim)
    plot.drug_effect(output_file)

    print('Fit pbpk 2-scan calculation time (mins): ', (time.time()-start)/60)


if __name__ == "__main__":
    main()