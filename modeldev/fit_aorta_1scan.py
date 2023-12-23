import os
import time
import pandas as pd
import numpy as np

import data
import plot
from models.aorta import AortaOneScan as Aorta


def read(data, params):

    print('Reading aorta...')
    aorta = Aorta(
        weight = data['weight'],
        dose = data['dose1'],
        tdce = data['time1'],
        tmolli = [data['T1time1'], data['T1time2']],
    )
    aorta.read_csv(params)
    return aorta


def fit_aorta(data):

    # Fit aorta
    print('Fitting aorta...')
    aorta = Aorta(
        weight = data['weight'],
        dose = data['dose1'],
        tdce = np.array(data['time1']),
        tmolli = [data['T1time1'], data['T1time2']],
        Sdce = data['aorta1'],
        R1molli = np.array([1000.0/data['T1aorta1'], 1000.0/data['T1aorta2']]),
        callback = False,
        ptol = 1e-3,
        dcevalid = data['aorta_valid1'],
    )
    # Perform fit
    
    aorta.estimate_p()
    print('Aorta goodness of fit: ', aorta.goodness())
    aorta.fit_p()
    print('Aorta goodness of fit: ', aorta.goodness())
    return aorta



def fit_data(datapath, resultspath):

    output_file = os.path.join(resultspath, 'parameters.csv')
    output = pd.DataFrame(columns=['subject','visit','structure','name','value','unit'])
    for visit in [f.name for f in os.scandir(datapath) if f.is_dir()]:
        visitdatapath = os.path.join(datapath, visit)
        for s in os.listdir(visitdatapath):
            subj = os.path.join(visitdatapath, s)
            print('Fitting ', visit, subj)
            subj_data = data.read(subj)
            aorta = fit_aorta(subj_data)
            aorta.to_csv(os.path.join(resultspath, s[:3] + '_' + visit + '.csv'))
            aorta.plot_fit(save=True, show=False, path=resultspath, prefix=s[:3] + '_' + visit)
            aorta_pars = aorta.export_p()
            aorta_pars['subject'] = s[:3]
            aorta_pars['visit'] = visit
            aorta_pars['structure'] = 'aorta'
            output = pd.concat([output, aorta_pars])
    try:
        output['parameter'] = output.index
        output.to_csv(output_file, index=False)
    except:
        print("Can't write to file ", output_file)
        print("Please close the file before saving data")



def main(data, results):

    start = time.time()

    resultspath = os.path.join(results, 'aorta_1scan')
    
    fit_data(data, resultspath)

    ylim = {}
    output_file = os.path.join(resultspath, 'parameters.csv')
    plot.create_bar_chart(output_file, ylim=ylim)

    print('Fit aorta 1-scan calculation time (mins): ', (time.time()-start)/60)


if __name__ == "__main__":
    main()