import numpy as np
import pandas as pd
import os
import nibabel as nib
import nrrd
from utilities import helper

def create_csv(info):
    subject, visit, scan = info['subject_details']

    r_paths = [results_path_v1, results_path_v2]
    with pd.ExcelWriter(f'{r_paths[visit-1]}\\00{subject}-visit{visit}.xlsx', engine='xlsxwriter') as writer:
        for scan in scans:
            aorta_signal = np.load(f"{output_path}\\aifs\\S{subject}_v{visit}_s{scan}_aif_postc.npz")['aif']
            liver_signal = np.load(f"{output_path}\\liver\\S{subject}_v{visit}_s{scan}_liver.npz")['liver']
            timepoints = np.load(f"{output_path}\\arrays\\{info['subject_path']}\\S{subject}_v{visit}_s{scan}_timepoints.npz")['timepoints']

        if save_signals:
            # Create a DataFrame with time and aif_signal as columns
            df = pd.DataFrame({'time': time, 'fa':12, 'liver': liver_signal, 'liver-2':'', 'aorta': aif_signal, 'portal-vein':'', 'kidney':'', 'noise':'', 'fat':'', 'spleen':'', 'vena-cava':'', 'gallbladder':'', 'left-ventricle':'', 'right-ventricle':'', 'muscle':'', 'lung':'', 'vertebra':'', 'intestine':''})
            
            df.to_excel(writer, sheet_name=f'dyn{scan}', index=False)
        print(f'Subject{subject}_Visit{visit}_Scan{scan} AIF saved')
   