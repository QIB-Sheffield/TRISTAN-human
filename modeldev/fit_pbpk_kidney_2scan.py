import os
import time
import pandas as pd
import numpy as np

import data
import plot
import fit_pbpk_aorta_2scan as fit_aorta

from models.pbpk_kidney_nephron import TwoScan as ModelOneDose
from models.pbpk_kidney_nephron_ns import TwoScan as ModelTwoDose

model_obj = {
    True: ModelOneDose,
    False: ModelTwoDose,
}

structure = 'kidney'
def read(data, params):
    print('Reading ' + structure + '...')
    kidney_model = model_obj[data['dose2'] == 0]
    result = kidney_model()
    result.read_csv(params)
    return result

def fit(data, aorta):

    print('Fitting kidney...')
    kidney_model = model_obj[data['dose2'] == 0]
    result = kidney_model(
        dt = aorta.dt,
        tmax = aorta.tmax,
        J_aorta = aorta.p.value.CO*aorta.cb/1000,
        tdce = np.concatenate([data['time1'], data['time2']]),
        t0 = [data['t0'], data['t0']+data['time2'][0]],
        BAT = [aorta.p.value.BAT1, aorta.p.value.BAT2],
        Sdce = np.concatenate([data['kidney1'], data['kidney2']]),
        R1molli = np.array([1000.0/data['T1kidney1'], 1000.0/data['T1kidney3']]),
        dcevalid = np.concatenate([data['kidney_valid1'], data['kidney_valid2']]),
        ptol = 1e-6,
        #tstop = aorta.p.value.BAT + 90,
        kidney_volume = 2*data['kidney_volume'], # estimated volume of both kidneys
    )
    result.estimate_p()
    print('Goodness of fit (%): ', result.goodness())
    result.fit_p()
    print('Goodness of fit (%): ', result.goodness())
    return result


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
            aorta = fit_aorta.read(subj_data, aortapars)
            aorta.predict_signal()
            result = fit(subj_data, aorta)
            result.to_csv(os.path.join(resultsfolder, s[:3] + '_' + visit + '.csv'))
            result.plot_fit(save=True, show=False, path=resultsfolder, prefix=s[:3] + '_' + visit + '_kidney')
            pars = result.export_p()
            pars['subject'] = s[:3]
            pars['visit'] = visit
            pars['structure'] = 'kidney'
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
    filepath = os.path.dirname(__file__)
    datapath = os.path.join(filepath, 'data')
    results = os.path.join(filepath, 'results')
    resultspath = os.path.join(results, 'pbpk_kidney_2scan')
    output_file = os.path.join(resultspath, 'parameters.csv')
    aortaresults = os.path.join(results, 'pbpk_aorta_2scan') 

    fit_data(datapath, output_file, aortaresults)

    ylim = {
        # 'Kbh': [0,10],
        # 'Khe': [0,150],
        # 'k_he': [0,30],
        # 've': [0,100],
        # 'k_bh': [0,4], 
    }

    plot.create_bar_chart(output_file, ylim=ylim)
    #plot.create_box_plot(output_file, ylim=ylim)
    plot.drug_effect(output_file)

    print('Calculation time (mins): ', (time.time()-start)/60)

if __name__ == "__main__":
    main()