import os
import time
import pandas as pd
import numpy as np

import data
import plot
from models.aorta_1scan import OneShotOneScan as Aorta


def read(data, params):

    (   time1, fa1, aorta1, liver1, portal1,
        aorta_valid1, liver_valid1, portal_valid1,
        T1time1, T1aorta1, T1liver1, T1portal1,
        T1time2, T1aorta2, T1liver2, T1portal2,
        weight, dose1, t0) = data

    print('Reading aorta...')
    aorta = Aorta()
    # Set constants
    aorta.weight = weight
    aorta.t0 = t0
    aorta.set_dose(dose1)
    aorta.set_x(time1, valid=aorta_valid1)
    aorta.set_y(aorta1)
    aorta.set_R10(T1time1, 1000.0/T1aorta1)
    aorta.set_R11(T1time2, 1000.0/T1aorta2)
    # Set options
    aorta.callback = False
    aorta.ptol = 1e-3
    aorta.dose_tolerance = 0.1
    aorta.dt = 0.5
    # Read fitted parameters
    aorta.read_csv(params)
    return aorta


def fit_aorta(data):

    (   time1, fa1, aorta1, liver1, portal1,
        aorta_valid1, liver_valid1, portal_valid1,
        T1time1, T1aorta1, T1liver1, T1portal1,
        T1time2, T1aorta2, T1liver2, T1portal2,
        weight, dose1, t0) = data

    # Fit aorta
    print('Fitting aorta...')
    aorta = Aorta()
    # Set data
    aorta.weight = weight
    aorta.t0 = t0
    aorta.set_dose(dose1)
    aorta.set_x(time1, valid=aorta_valid1)
    aorta.set_y(aorta1)
    aorta.set_R10(T1time1, 1000.0/T1aorta1)
    aorta.set_R11(T1time2, 1000.0/T1aorta2)
    # Set fit options
    aorta.callback = False
    aorta.ptol = 1e-3
    aorta.dose_tolerance = 0.1
    aorta.dt = 0.5
    # Perform fit
    aorta.estimate_p()
    print('Aorta goodness of fit: ', aorta.goodness())
    aorta.fit_p()
    print('Aorta goodness of fit: ', aorta.goodness())
    return aorta



def fit_data(datapath, output_file):

    resultsfolder = os.path.dirname(output_file)
    output = pd.DataFrame(columns=['subject','visit','structure','name','value','unit'])
    for visit in ['baseline', 'rifampicin']:
        visitdatapath = os.path.join(datapath, visit)
        for s in os.listdir(visitdatapath):
            subj = os.path.join(visitdatapath, s)
            print('Fitting ', subj)
            subj_data = data.oneshot_onescan(subj)
            aorta = fit_aorta(subj_data)
            aorta.to_csv(os.path.join(resultsfolder, s[:3] + '_' + visit + '.csv'))
            aorta.plot_fit(save=True, show=False, path=resultsfolder, prefix=s[:3] + '_' + visit)
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

    return output_file



if __name__ == "__main__":

    start = time.time()

    #filepath = os.path.abspath("")
    filepath = os.path.dirname(__file__)
    datapath = os.path.join(filepath, 'data')
    resultspath = os.path.join(filepath, 'results')
    output_file = os.path.join(resultspath, 'parameters.csv')

    fit_data(datapath, output_file)

    ylim = {}
    plot.create_bar_chart(output_file, ylim=ylim)
    plot.create_box_plot(output_file, ylim=ylim)
    plot.drug_effect(output_file)

    print('Calculation time (mins): ', (time.time()-start)/60)