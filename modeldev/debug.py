import os
import time
import pandas as pd
import numpy as np

import data
import plot
import fit_aorta

from models.liver_cal_hf2cm_var2k import TwoShotTwoScan as Liver

filepath = os.path.abspath("")
datapath = os.path.join(filepath, 'devdata')
resultspath = os.path.join(filepath, 'devresults')
output_file = os.path.join(resultspath, 'parameters.csv')
resultsfolder = os.path.dirname(output_file)
s = os.listdir(datapath)[0]
subj = os.path.join(datapath, s)
subj_data = data.twoshot_twoscan(subj)
(   time1, fa1, aorta1, liver1, portal1,
    aorta_valid1, liver_valid1, portal_valid1,
    time2, fa2, aorta2, liver2, portal2,
    aorta_valid2, liver_valid2, portal_valid2,
    T1time1, T1aorta1, T1liver1, T1portal1,
    T1time2, T1aorta2, T1liver2, T1portal2,
    T1time3, T1aorta3, T1liver3, T1portal3, 
    weight, dose1, dose2) = subj_data

visit = 'baseline'
aortaresults = os.path.join(filepath, 'results_aorta') 
aortapars = os.path.join(aortaresults, s[:3] + '_' + visit + '.csv')
aorta = fit_aorta.read(subj_data, aortapars)


# Fit liver
liver = Liver(aorta)
# Set data
xvalid = np.concatenate([liver_valid1, liver_valid2, np.full(3, True)])
liver.set_x(time1, time2, [T1time1,T1time2,T1time3], valid=xvalid)
liver.set_y(liver1, liver2, [1000.0/T1liver1, 1000.0/T1liver2, 1000.0/T1liver3])
# Set fit options
liver.set_weights(0.1)
liver.ptol = 1e-6

print(liver.x.shape)
print(liver.valid[0].shape)
print(liver.xind.shape)
print(liver.x[liver.valid].shape)
print(liver.xf)