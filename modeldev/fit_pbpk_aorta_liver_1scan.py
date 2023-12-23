import os
import time
import pandas as pd
import numpy as np

import data
import plot
import fit_pbpk_aorta_1scan
import fit_pbpk_liver_1scan
from models.pbpk_aorta_liver import OneScan as Model

filepath = os.path.dirname(__file__)
results = os.path.join(filepath, 'results')


def read(data, params):
    structure = 'pbpk'
    print('Reading ' + structure + '...')
    result = Model(
        weight = data['weight'],
        dose = data['dose1'], 
        tdce = np.array(data['time1']),   
    )
    result.read_csv(params)
    return result


def fit(data, ap, lp): 
    structure = 'pbpk'
    print('Fitting ' + structure + '...')
    result = Model(
        weight = data['weight'],
        dose = data['dose1'],   
        tdce = np.array(data['time1']),   
        Sdce = [
            np.array(data['aorta1']),
            np.array(data['liver1']),
        ],
        R1molli = [
            1000.0/data['T1aorta1'],
            1000.0/data['T1liver1'],
        ],
        callback = False,
        ptol = 1e-3,
        dcevalid = [
            np.array(data['aorta_valid1']),
            np.array(data['liver_valid1']),
        ],
        liver_volume = data['liver_volume'],
        # Initial values
        aorta = ap,
        liver = lp,
    )
    E_k = (ap.at['E_b','value'] - lp.at['FF_l','value']*lp.at['E_l','value'])/(1-lp.at['FF_l','value'])
    if E_k>=result.p.at['E_k','lower bound'] and E_k<=result.p.at['E_k','upper bound']:
        result.p.at['E_k','value'] = E_k
    #result.estimate_S0()
    print(structure + ' goodness of fit: ', result.goodness())
    result.fit_p()
    print(structure + ' goodness of fit: ', result.goodness())
    return result


def fit_data(datapath, output_file):
    structure = 'pbpk'
    resultsfolder = os.path.dirname(output_file)
    output = pd.DataFrame(columns=['subject','visit','structure','name','value','unit'])
    for visit in ['baseline', 'rifampicin']:
        visitdatapath = os.path.join(datapath, visit)
        for s in os.listdir(visitdatapath):
            subj = os.path.join(visitdatapath, s)
            print('Fitting ', subj)
            subj_data = data.read(subj)
            inputpars = os.path.join(results, 'pbpk_aorta_1scan', s[:3] + '_' + visit + '.csv')
            aorta = fit_pbpk_aorta_1scan.read(subj_data, inputpars)
            inputpars = os.path.join(results, 'pbpk_liver_1scan', s[:3] + '_' + visit + '.csv')
            liver = fit_pbpk_liver_1scan.read(subj_data, inputpars)
            result = fit(subj_data, aorta.p, liver.p)
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
    datapath = os.path.join(filepath, 'data')
    resultspath = os.path.join(results, 'pbpk_aorta_liver_1scan')
    output_file = os.path.join(resultspath, 'parameters.csv')

    fit_data(datapath, output_file)

    ylim = {}
    plot.create_bar_chart(output_file, ylim=ylim)
    plot.drug_effect(output_file)

    print('Fit pbpk 1-scan calculation time (mins): ', (time.time()-start)/60)


if __name__ == "__main__":
    main()