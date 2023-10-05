import os
import time
import pandas as pd
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

import data
import plot
import fit_aorta

from models.liver_cal_hf2cm import TwoShotTwoScan as LiverOneDose
from models.liver_cal_hf2cm_var22 import TwoShotTwoScan as Liver


#wcal = 1e-2
ptol = 1e-9

def fit_liver(data, aorta):

    (   time1, fa1, aorta1, liver1, portal1,
        aorta_valid1, liver_valid1, portal_valid1,
        time2, fa2, aorta2, liver2, portal2,
        aorta_valid2, liver_valid2, portal_valid2,
        T1time1, T1aorta1, T1liver1, T1portal1,
        T1time2, T1aorta2, T1liver2, T1portal2,
        T1time3, T1aorta3, T1liver3, T1portal3,
        weight, dose1, dose2, t0) = data

    # Fit liver
    print('Fitting liver...')
    if dose2 == 0:
        liver = LiverOneDose(aorta)
    else:
        liver = Liver(aorta)
    # Set data
    xvalid = np.concatenate([liver_valid1, liver_valid2, np.full(3, True)])
    liver.set_x(time1, time2, [T1time1,T1time2,T1time3], valid=xvalid)
    liver.set_y(liver1, liver2, [1000.0/T1liver1, 1000.0/T1liver2, 1000.0/T1liver3])
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
            subj_data = data.twoshot_twoscan(subj)
            aortapars = os.path.join(aortaresults, s[:3] + '_' + visit + '.csv')
            aorta = fit_aorta.read(subj_data, aortapars)
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
    aortaresults = os.path.join(filepath, 'results_aorta') 

    fit_data(datapath, output_file, aortaresults)

    ylim = {
        'Kbh': [0,5],
        'Kbh_i': [0,5],
        'Kbh_f': [0,5],
        'Khe': [0,150],
        'k_he': [0,30],
        'k_he_i': [0,30],
        'k_he_m': [0,30],
        'k_he_f': [0,30],
        've': [0,100],
        'k_bh': [0,4], 
    }

    plot.calc_effect_size(output_file)
    plot.pivot_table(output_file)

    # Primary parameters
    report_file = os.path.join(resultspath, '_report.pdf')
    p = PdfPages(report_file)
    
    fig = plot.drug_effect_function(output_file)
    #fig.savefig(p, format='pdf') 
    p.savefig()
    fig = plot.line_plot_function(output_file)
    #fig.savefig(p, format='pdf') 
    p.savefig()
    fig = plot.diurnal_k(output_file)
    #fig.savefig(p, format='pdf') 
    p.savefig()
    p.close()


    # Secondary parameters
    plot.line_plot_extracellular(output_file)
    plot.create_bar_chart(output_file, ylim=ylim)
    plot.create_box_plot(output_file, ylim=ylim)
    plot.drug_effect(output_file)


    print('Calculation time (mins): ', (time.time()-start)/60)