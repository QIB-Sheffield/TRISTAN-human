import os
import time
import pandas as pd
import numpy as np

import data
import plot

import fit_aorta_2scan

from models.kidney_2cfm import KidneyOneScan, KidneyTwoScan

model_obj = {
    True: KidneyOneScan,
    False: KidneyTwoScan,
}

def fit_kidney(data, aorta):
    # Fit kidney
    organ = 'kidney'
    print('Fitting '+organ+'...')
    tissue_model = model_obj[data['dose2'] == 0]
    tissue = tissue_model(
        dt = aorta.dt,
        tmax = aorta.tmax,
        cb = aorta.cb,
        tdce = np.concatenate([data['time1'], data['time2']]),
        t0 = [data['t0'], data['t0']+data['time2'][0]],
        tmolli = [data['T1time1'], data['T1time2'], data['T1time3']],  
        BAT = [aorta.p.value.BAT1, aorta.p.value.BAT2],
        Sdce = np.concatenate([data[organ+'1'], data[organ+'2']]),
        R1molli = np.array([1000.0/data['T1'+organ+'1'], 1000.0/data['T1'+organ+'2'], 1000.0/data['T1'+organ+'3']]),
        dcevalid = np.concatenate([data[organ+'_valid1'], data[organ+'_valid2']]),
        ptol = 1e-6,            
    )
    tissue.estimate_p()
    print('Goodness of fit (%): ', tissue.goodness())
    tissue.fit_p()
    print('Goodness of fit (%): ', tissue.goodness())
    return tissue


def fit_data(datapath, output_file, aortaresults_1scan, aortaresults_2scan):

    resultsfolder = os.path.dirname(output_file)
    output = pd.DataFrame(columns=['subject','visit','structure','name','value','unit'])
    for visit in ['baseline', 'rifampicin']:
        visitdatapath = os.path.join(datapath, visit)
        for s in os.listdir(visitdatapath):
            subj = os.path.join(visitdatapath, s)
            print('Fitting ', subj)
            subj_data = data.read(subj)
            aortapars = os.path.join(aortaresults_2scan, s[:3] + '_' + visit + '.csv')
            aorta = fit_aorta_2scan.read(subj_data, aortapars)
            aorta.predict_signal()
            kidney = fit_kidney(subj_data, aorta)
            kidney.to_csv(os.path.join(resultsfolder, s[:3] + '_' + visit + '.csv'))
            kidney.plot_fit(save=True, show=False, path=resultsfolder, prefix=s[:3] + '_' + visit + '_Kidney')
            kidney_pars = kidney.export_p()
            kidney_pars['subject'] = s[:3]
            kidney_pars['visit'] = visit
            kidney_pars['structure'] = 'kidney'
            output = pd.concat([output, kidney_pars])
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
    resultspath = os.path.join(results, 'kidney_2scan')
    output_file = os.path.join(resultspath, 'parameters.csv')
    aortaresults_1scan = os.path.join(results, 'aorta_1scan') 
    aortaresults_2scan = os.path.join(results, 'aorta_2scan') 

    fit_data(datapath, output_file, aortaresults_1scan, aortaresults_2scan)

    ylim = {}
    plot.create_bar_chart(output_file, ylim=ylim)
    plot.create_box_plot(output_file, ylim=ylim)
    plot.drug_effect(output_file)

    print('Calculation time (mins): ', (time.time()-start)/60)


if __name__ == "__main__":
    main()