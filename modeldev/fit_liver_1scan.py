import os
import time
import pandas as pd
import numpy as np

import data
import plot
import fit_aorta_1scan

from models.liver_cal_hf2cm_1scan import OneShotOneScan as Liver


#wcal = 1e-2
ptol = 1e-6

def fit_liver(data, aorta):

    (   time1, fa1, aorta1, liver1, portal1,
        aorta_valid1, liver_valid1, portal_valid1,
        T1time1, T1aorta1, T1liver1, T1portal1,
        T1time2, T1aorta2, T1liver2, T1portal2,
        weight, dose1, t0) = data

    # Fit liver
    print('Fitting liver...')
    liver = Liver(aorta)
    # Set data
    xvalid = np.concatenate([liver_valid1, np.full(2, True)])
    liver.set_x(time1, [T1time1,T1time2], valid=xvalid)
    liver.set_y(liver1, [1000.0/T1liver1, 1000.0/T1liver2])
    #liver.set_cweight(wcal)
    # Set fit options
    liver.ptol = ptol
    # Estimate parameters from data
    liver.estimate_p()
    print('Goodness of fit (%): ', liver.goodness())
    liver.fit_p()
    print('Goodness of fit (%): ', liver.goodness())
    return liver



def fit_data(datapath, output_file, aortaresults):

    resultsfolder = os.path.dirname(output_file)
    output = pd.DataFrame(columns=['subject','visit','structure','name','value','unit'])
    for visit in ['baseline', 'rifampicin']:
        visitdatapath = os.path.join(datapath, visit)
        for s in os.listdir(visitdatapath):
            subj = os.path.join(visitdatapath, s)
            print('Fitting ', subj)
            subj_data = data.oneshot_onescan(subj)
            aortapars = os.path.join(aortaresults, s[:3] + '_' + visit + '.csv')
            aorta = fit_aorta_1scan.read(subj_data, aortapars)
            liver = fit_liver(subj_data, aorta)
            liver.to_csv(os.path.join(resultsfolder, s[:3] + '_' + visit + '.csv'))
            liver.plot_fit(save=True, show=False, path=resultsfolder, prefix=s[:3] + '_' + visit + '_Liver')
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



if __name__ == "__main__":

    start = time.time()

    #filepath = os.path.abspath("")
    filepath = os.path.dirname(__file__)
    datapath = os.path.join(filepath, 'data')
    resultspath = os.path.join(filepath, 'results')
    output_file = os.path.join(resultspath, 'parameters.csv')
    aortaresults = os.path.join(filepath, 'results_aorta_1scan') 

    fit_data(datapath, output_file, aortaresults)

    ylim = {
        'Kbh': [0,5],
        'Khe': [0,150],
        'k_he': [0,30],
        've': [0,100],
        'k_bh': [0,4], 
    }

    plot.create_bar_chart(output_file, ylim=ylim)
    plot.create_box_plot(output_file, ylim=ylim)
    plot.drug_effect(output_file)

    print('Calculation time (mins): ', (time.time()-start)/60)