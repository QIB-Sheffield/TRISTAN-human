import os
import time
import pandas as pd
import numpy as np

import data
import plot
from models.aorta import TwoShotTwoScan as Aorta

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

# Fit aorta
aorta = Aorta()
# Set data
aorta.weight = weight
aorta.set_dose(dose1, dose2)
aorta.set_x(time1, time2, valid=np.append(aorta_valid1, aorta_valid2))
aorta.set_y(aorta1, aorta2)
aorta.set_R10(T1time1, 1000.0/T1aorta1)
aorta.set_R11(T1time2, 1000.0/T1aorta2)
aorta.set_R12(T1time3, 1000.0/T1aorta3)
# Set fit options
aorta.callback = False
aorta.ptol = 1e-6
aorta.dose_tolerance = 0.1
aorta.dt = 0.5

aorta.estimate_p()
aorta.plot_fit()