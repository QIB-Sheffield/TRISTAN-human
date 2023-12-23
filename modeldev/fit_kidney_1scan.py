import os
import time
import pandas as pd
import numpy as np

import data
import plot
import fit_aorta_1scan

#from models.kidney_reabs_1scan import Kidney
from models.kidney_2cfm import KidneyOneScan as Kidney

def fit_kidney(data, aorta):

    print('Fitting kidney...')
    tissue = Kidney(
        dt = aorta.dt,
        tmax = aorta.tmax,
        cb = aorta.cb,
        tdce = data['time1'],
        tmolli = [data['T1time1'], data['T1time2']],
        BAT = aorta.p.value.BAT,
        Sdce = data['kidney1'],
        R1molli = np.array([1000.0/data['T1kidney1'], 1000.0/data['T1kidney2']]),
        dcevalid = data['kidney_valid1'],
        cweight = None,
        ptol = 1e-6,
    )
    tissue.estimate_p()
    print('Goodness of fit (%): ', tissue.goodness())
    tissue.fit_p()
    print('Goodness of fit (%): ', tissue.goodness())
    return tissue



def fit_data(datapath, output_file, aortaresults):

    resultsfolder = os.path.dirname(output_file)
    output = pd.DataFrame(columns=['subject','visit','structure','name','value','unit'])
    for visit in ['baseline', 'rifampicin']:
        visitdatapath = os.path.join(datapath, visit)
        for s in os.listdir(visitdatapath):
            subj = os.path.join(visitdatapath, s)
            print('Fitting ', subj)
            subj_data = data.read(subj)
            aortapars = os.path.join(aortaresults, s[:3] + '_' + visit + '.csv')
            aorta = fit_aorta_1scan.read(subj_data, aortapars)
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
    resultspath = os.path.join(results, 'kidney_1scan_2cum')
    output_file = os.path.join(resultspath, 'parameters.csv')
    aortaresults = os.path.join(results, 'aorta_1scan') 

    fit_data(datapath, output_file, aortaresults)

    ylim = {}
    plot.create_bar_chart(output_file, ylim=ylim)
    plot.create_box_plot(output_file, ylim=ylim)
    plot.drug_effect(output_file)

    print('Calculation time (mins): ', (time.time()-start)/60)


if __name__ == "__main__":
    main()