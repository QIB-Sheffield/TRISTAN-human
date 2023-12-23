import os
import time
import pandas as pd
import numpy as np

import data
import plot
import fit_aorta_1scan

from models.liver_2cm import LiverOneScan as Tissue

def fit_liver(data, aorta):

    print('Fitting liver...')
    liver = Tissue(
        dt = aorta.dt,
        tmax = aorta.tmax,
        cb = aorta.cb,
        tdce = data['time1'],
        tmolli = [data['T1time1'], data['T1time2']],
        BAT = aorta.p.value.BAT,
        Sdce = data['liver1'],
        R1molli = np.array([1000.0/data['T1liver1'], 1000.0/data['T1liver2']]),
        dcevalid = data['liver_valid1'],
        cweight = None,
        ptol = 1e-6,
        liver_volume = data['liver_volume'],
    )
    liver.estimate_p()
    print('Goodness of fit (%): ', liver.goodness())
    liver.fit_p()
    print('Goodness of fit (%): ', liver.goodness())
    return liver


def fit_data(datapath, resultspath, aortaresults):

    output_file = os.path.join(resultspath, 'parameters.csv')
    output = pd.DataFrame(columns=['subject','visit','structure','name','value','unit'])
    for visit in [f.name for f in os.scandir(datapath) if f.is_dir()]:
        visitdatapath = os.path.join(datapath, visit)
        for s in os.listdir(visitdatapath):
            subj = os.path.join(visitdatapath, s)
            print('Fitting ', subj)
            subj_data = data.read(subj)
            aortapars = os.path.join(aortaresults, s[:3] + '_' + visit + '.csv')
            aorta = fit_aorta_1scan.read(subj_data, aortapars)
            aorta.predict_signal()
            liver = fit_liver(subj_data, aorta)
            liver.to_csv(os.path.join(resultspath, s[:3] + '_' + visit + '.csv'))
            liver.plot_fit(save=True, show=False, path=resultspath, prefix=s[:3] + '_' + visit + '_Liver')
            liver_pars = liver.export_p()
            liver_pars['subject'] = s[:3]
            liver_pars['visit'] = visit
            liver_pars['structure'] = 'liver'
            output = pd.concat([output, liver_pars])
    try:
        output['parameter'] = output.index
        output.to_csv(output_file, index=False)
    except:
        print("Can't write to file ", output_file)
        print("Please close the file before saving data")

    return output_file



def main(data, results):

    start = time.time()

    resultspath = os.path.join(results, 'liver_1scan')
    aortaresults = os.path.join(results, 'aorta_1scan') 
   
    fit_data(data, resultspath, aortaresults)

    output_file = os.path.join(resultspath, 'parameters.csv')
    plot.create_bar_chart(output_file)

    print('Calculation time (mins): ', (time.time()-start)/60)


if __name__ == "__main__":
    main()