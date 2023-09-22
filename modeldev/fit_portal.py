import os
import time
import pandas as pd
import numpy as np

import data
import plot
from models.aorta import TwoShotTwoScan as Aorta
from models.portal_vein import TwoShotTwoScan as Portal


def fit_portal(data):

    (   time1, fa1, aorta1, liver1, portal1,
        aorta_valid1, liver_valid1, portal_valid1,
        time2, fa2, aorta2, liver2, portal2,
        aorta_valid2, liver_valid2, portal_valid2,
        T1time1, T1aorta1, T1liver1, T1portal1,
        T1time2, T1aorta2, T1liver2, T1portal2,
        T1time3, T1aorta3, T1liver3, T1portal3, 
        weight, dose1, dose2) = data

    aorta = Aorta()
    # Set data
    aorta.weight = weight
    aorta.set_dose(dose1, dose2)
    aorta.set_R10(T1time1, 1000.0/T1aorta1)
    aorta.set_R11(T1time2, 1000.0/T1aorta2)
    aorta.set_R12(T1time3, 1000.0/T1aorta3)
    aorta.set_x(time1, time2, valid=np.append(aorta_valid1, aorta_valid2))
    aorta.set_y(aorta1, aorta2)
    # Set fit options
    aorta.callback = True
    aorta.ptol = 1e-3
    aorta.dose_tolerance = 0.1
    aorta.dt = 0.5
    # Perform fit
    aorta.estimate_p()
    aorta.fit_p()

    portal = Portal(aorta)
    # Set data
    portal.set_R10(T1time1, 1000.0/T1portal1)
    portal.set_R11(T1time2, 1000.0/T1portal2)
    portal.set_R12(T1time3, 1000.0/T1portal3)
    portal.set_x(time1, time2, valid=np.append(portal_valid1, portal_valid2))
    portal.set_y(portal1, portal2)
    # Set fit options
    portal.callback = True
    portal.ptol = 1e-3
    # Perform fit
    portal.estimate_p()
    portal.fit_p()

    return aorta, portal


def fit_data(datafolder, output_file):

    resultsfolder = os.path.dirname(output_file)
    output = pd.DataFrame(columns=['subject','visit','structure','name','value','unit'])
    
    for visit in ['rifampicin','baseline']:

        datapath = os.path.join(datafolder, visit)
        for s in os.listdir(datapath):

            subj = os.path.join(datapath, s)
            subj_data = data.twoshot_twoscan(subj) 
            aorta, portal = fit_portal(subj_data)
            aorta.plot_fit(save=True, show=False, path=resultsfolder, prefix=s[:3] + '_' + visit + '_Aorta')
            portal.plot_fit(save=True, show=False, path=resultsfolder, prefix=s[:3] + '_' + visit + '_Portal')
            aorta_pars = aorta.export_p()
            portal_pars = portal.export_p()
            aorta_pars['subject'] = s[:3]
            portal_pars['subject'] = s[:3]
            aorta_pars['visit'] = visit
            portal_pars['visit'] = visit
            aorta_pars['structure'] = 'aorta'
            portal_pars['structure'] = 'portal_vein'
            output = pd.concat([output, aorta_pars, portal_pars])

    output['parameter'] = output.index
    try:
        output.to_csv(output_file, index=False)
    except:
        print("Can't write to file ", output_file)
        print("Please close the file before saving data")

    return output_file



if __name__ == "__main__":

    start = time.time()

    filepath = os.path.dirname(__file__)
    datafolder = os.path.join(filepath, 'data')
    resultsfolder = os.path.join(filepath, 'portal_results')
    output_file = os.path.join(resultsfolder, 'portal_parameters.csv')

    fit_data(datafolder, output_file)
    plot.create_bar_chart(output_file)

    print('Calculation time (mins): ', (time.time()-start)/60)