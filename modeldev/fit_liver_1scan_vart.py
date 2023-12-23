import os
import time
import pandas as pd
import numpy as np

import data
import plot
import fit_aorta_1scan

from models.liver_1scan import Liver


#wcal = 1e-2
ptol = 1e-6

def fit_liver(data, aorta, tmax):

    print('Fitting liver...')
    liver = Liver(
        dt = aorta.dt,
        tmax = aorta.tmax,
        cb = aorta.cb,
        field_strength = aorta.field_strength,
        TR = aorta.TR,
        FA = aorta.p.value.FA,
        tdce = data['time1'],
        tmolli = [data['T1time1'], data['T1time2']],
        BAT = aorta.p.value.BAT,
        Sdce = data['liver1'],
        R1molli = np.array([1000.0/data['T1liver1'], 1000.0/data['T1liver2']]),
        dcevalid = data['liver_valid1'],
        tstop = tmax,
        cweight = None,
        ptol = 1e-6,
    )
    liver.estimate_p()
    print('Goodness of fit (%): ', liver.goodness())
    liver.fit_p()
    print('Goodness of fit (%): ', liver.goodness())
    return liver



def fit_data(datapath, output_file, aortaresults):

    resultsfolder = os.path.dirname(output_file)
    output = pd.DataFrame(columns=['subject','visit','structure','name','value','unit','tmax'])
    for visit in ['baseline', 'rifampicin']:
        visitdatapath = os.path.join(datapath, visit)
        for s in os.listdir(visitdatapath):
            subj = os.path.join(visitdatapath, s)
            subj_data = data.read(subj)
            aortapars = os.path.join(aortaresults, s[:3] + '_' + visit + '.csv')
            aorta = fit_aorta_1scan.read(subj_data, aortapars)
            aorta.predict_signal()
            all_tmax = np.arange(aorta.p.value.BAT+60, subj_data['time1'][-1], 60)
            for tmax in all_tmax:
                print('Fitting ', visit, s[:3], tmax)
                tmax_str = str(np.round(tmax/60))
                liver = fit_liver(subj_data, aorta, tmax)
                liver.to_csv(os.path.join(resultsfolder, s[:3] + '_' + visit + '_' + tmax_str +'.csv'))
                liver.plot_fit(save=True, show=False, path=resultsfolder, prefix=s[:3] + '_' + visit + '_' + tmax_str + '_Liver')
                liver_pars = liver.export_p()
                liver_pars['subject'] = s[:3]
                liver_pars['visit'] = visit
                liver_pars['structure'] = 'liver'
                liver_pars['tmax'] = tmax
                output = pd.concat([output, liver_pars])
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
    resultspath = os.path.join(results, 'liver_1scan_vart')
    output_file = os.path.join(resultspath, 'parameters.csv')
    aortaresults = os.path.join(results, 'aorta_1scan') 

    fit_data(datapath, output_file, aortaresults)

    print('Calculation time (mins): ', (time.time()-start)/60)

if __name__ == "__main__":
    main()