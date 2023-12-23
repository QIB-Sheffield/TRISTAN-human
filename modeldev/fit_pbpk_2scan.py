import os
import time
import pandas as pd
import numpy as np

import data
import plot
import fit_pbpk_aorta_2scan
import fit_pbpk_liver_2scan
import fit_pbpk_kidney_2scan

from models.pbpk import TwoScan as ModelOneDose
from models.pbpk_nsl import TwoScan as ModelTwoDose

model_obj = {
    True: ModelOneDose,
    False: ModelTwoDose,
}

filepath = os.path.dirname(__file__)
results = os.path.join(filepath, 'results')

def read(data, params):
    structure = 'pbpk'
    print('Reading ' + structure + '...')
    liver_model = model_obj[data['dose2'] == 0]
    result = liver_model()
    result.read_csv(params)
    return result


def fit(data, ap, lp, kp):
    structure = 'pbpk'
    print('Fitting ' + structure + '...')
    liver_model = model_obj[data['dose2'] == 0]
    result = liver_model(
        weight = data['weight'],
        dose = [data['dose1'], data['dose2']],   
        tdce = np.concatenate([data['time1'], data['time2']]),  
        t0 = [data['t0'], data['t0']+data['time2'][0]],
        Sdce = [
            np.concatenate([data['aorta1'], data['aorta2']]),
            np.concatenate([data['liver1'], data['liver2']]),
            np.concatenate([data['kidney1'], data['kidney2']]),
        ],
        R1molli = [
            np.array([1000.0/data['T1aorta1'], 1000.0/data['T1aorta3']]),
            np.array([1000.0/data['T1liver1'], 1000.0/data['T1liver3']]),
            np.array([1000.0/data['T1kidney1'], 1000.0/data['T1kidney3']]),
        ],
        callback = False,
        ptol = 1e-3,
        dcevalid = [
            np.concatenate([data['aorta_valid1'], data['aorta_valid2']]),
            np.concatenate([data['liver_valid1'], data['liver_valid2']]),
            np.concatenate([data['kidney_valid1'], data['kidney_valid2']]),
        ],
        #tstop = [None,None,ap.value.BAT1+90],
        liver_volume = data['liver_volume'],
        kidney_volume = data['kidney_volume'],
        # Initial values
        aorta = ap,
        liver = lp,
        kidney = kp,
    )
    if data['dose2'] == 0:
        E_l = lp.at['E_l','value']
    else:
        E_li = lp.at['E_li','value']
        E_lf = lp.at['E_li','value']
        E_l = (E_li+E_lf)/2
    E_k = (ap.at['E_b','value']-lp.at['FF_l','value']*E_l) / (1-lp.at['FF_l','value'])
    F_k = E_k/(1-E_k)
    if F_k>=result.p.at['F_k','lower bound'] and F_k<=result.p.at['F_k','upper bound']:
        result.p.at['F_k','value'] = F_k
    #result.estimate_p()
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
            inputpars = os.path.join(results, 'pbpk_aorta_2scan', s[:3] + '_' + visit + '.csv')
            aorta = fit_pbpk_aorta_2scan.read(subj_data, inputpars)
            inputpars = os.path.join(results, 'pbpk_liver_2scan', s[:3] + '_' + visit + '.csv')
            liver = fit_pbpk_liver_2scan.read(subj_data, inputpars)
            inputpars = os.path.join(results, 'pbpk_kidney_2scan', s[:3] + '_' + visit + '.csv')
            kidney = fit_pbpk_kidney_2scan.read(subj_data, inputpars)
            result = fit(subj_data, aorta.p, liver.p, kidney.p)
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
    resultspath = os.path.join(results, 'pbpk_2scan')
    output_file = os.path.join(resultspath, 'parameters.csv')
    
    fit_data(datapath, output_file)

    ylim = {}
    plot.create_bar_chart(output_file, ylim=ylim)
    plot.create_box_plot(output_file, ylim=ylim)
    plot.drug_effect(output_file)

    print('Fit pbpk 2-scan calculation time (mins): ', (time.time()-start)/60)


if __name__ == "__main__":
    main()