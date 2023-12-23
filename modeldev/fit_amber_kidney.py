import os
import time
import pandas as pd
import numpy as np

import data
import plot
import fit_amber_aorta as fit_aorta

from models.pbpk_kidney_nephron_short import OneScan

structure = 'kidney'

def read(time, params):
    print('Reading ' + structure + '...')
    result = OneScan(
        TR = 4.80/1000.0, #sec
        FA = 17.0, #deg
        tdce = time,  
    )
    result.read_csv(params)
    return result

def fit(time, signal, R1, kidney_volume, aorta):

    print('Fitting kidney...')
    result = OneScan(
        agent = 'Dotarem',
        TR = 4.80/1000.0, #sec
        FA = 17.0, #deg
        dt = aorta.dt,
        tmax = aorta.tmax,
        field_strength = 3.0,      
        Hct = 0.36,  
        J_aorta = aorta.p.value.CO*aorta.cb/1000,
        tdce = time,
        BAT = aorta.p.value.BAT,
        Sdce = signal,
        R1molli = R1,
        ptol = 1e-6,
        #tstop = aorta.p.value.BAT + 90,
        kidney_volume = kidney_volume, 
        CO = aorta.p.value.CO,
    )
    result.estimate_p()
    print('Goodness of fit (%): ', result.goodness())
    result.fit_p()
    print('Goodness of fit (%): ', result.goodness())
    return result


def fit_data(datadir, aorta_results, kidney_results):
    datafile = os.path.join(datadir, 'Overview.xlsx')
    parfile = os.path.join(datadir, 'ROI size.xlsx')
    data = pd.read_excel(datafile, sheet_name='Blad1')
    const = pd.read_excel(parfile, sheet_name='Blad1')
    weight = pd.read_excel(parfile, sheet_name='Blad2')
    output = pd.DataFrame(columns=['subject','visit','structure','name','value','unit'])
    goodness = []
    for s in ['1','2','3','4']:
        for visit in ['1','2']:
            for kid in ['LK','RK']:
                print('Fitting ', s, visit, kid)
                curve = 'Dog' + s + '.' + visit + ' ' + kid
                wght = weight[weight.Dog==int(s)].weight.values[0]
                time = data['Time'].values
                aortapars = os.path.join(aorta_results, 'Dog' + s + '.' + visit + ' AIF.csv')
                aorta = fit_aorta.read(time, wght, aortapars)
                aorta.predict_signal()
                c = const[(const.Dog==int(s)) & (const.Session==int(visit)) & (const.Kidney==kid)]
                kidney_volume = c['ROI size (ml)'].values[0]
                R1 = 1000/c['T1 Kidney'].values[0]
                result = fit(time, data[curve].values, R1, kidney_volume, aorta)
                result.to_csv(os.path.join(kidney_results, s[:3] + '_' + kid + '_' + visit + '.csv'))
                result.plot_fit(save=True, show=False, path=kidney_results, prefix=s[:3] + '_' + kid + '_' + visit)
                pars = result.export_p()
                pars['subject'] = s[:3]
                pars['visit'] = visit
                pars['structure'] = kid
                output = pd.concat([output, pars])
                goodness.append(result.goodness())
    try:
        output['parameter'] = output.index
        output.to_csv(os.path.join(kidney_results, 'parameters.csv'), index=False)
        output.to_pickle(os.path.join(kidney_results, 'parameters.pkl'))
    except:
        print("Can't write to parameter file ")
        print("Please close the file before saving data")

    return goodness



def main(datadir, aorta_results, kidney_results):

    start = time.time()

    goodness = fit_data(datadir, aorta_results, kidney_results)

    output_file = os.path.join(kidney_results, 'parameters.csv')
    plot.create_bar_chart(output_file, ylim={})

    print('Calculation time (mins): ', (time.time()-start)/60)
    print('Goodness of fit (%) - min', np.amin(goodness))
    print('Goodness of fit (%) - mean', np.mean(goodness))
    print('Goodness of fit (%) - max', np.amax(goodness))

if __name__ == "__main__":
    main()