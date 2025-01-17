import numpy as np
import pandas as pd
import os
import nibabel as nib
from utilities import helper

def create_csv(info):

    logger = info['logger']

    logger.info('THIS FUNCTION IS NOT YET FULLY IMPLEMENTED AND WILL NOT WORK OR RETURN DEFAULT VALUES PLEASE SEE THE COMMENTS IN THE CODE')
    print('THIS FUNCTION IS NOT YET FULLY IMPLEMENTED AND WILL NOT WORK OR RETURN DEFAULT VALUES  PLEASE SEE THE COMMENTS IN THE CODE')

    subject, visit = info['subject'], info['visit']
    
    output_path = info['output_path']
    results_path = info['results_path']
    xlsx_file = f'00{subject}-visit{visit}.xlsx'

    with pd.ExcelWriter(os.path.join(results_path, xlsx_file), engine='xlsxwriter') as writer:
        for scan in [1]:
            aorta_signal = np.load(os.path.join(output_path, 'aif', f'S{subject}_v{visit}_s{scan}_aif_prec.npz'))['aif']
            liver_signal = np.load(os.path.join(output_path, 'liver', f'S{subject}_v{visit}_s{scan}_liver_postc.npz'))['liver']
            data = np.load(os.path.join(info['output_path'], 'arrays', info['subject_path'], 'combined_dynamic.npz'), allow_pickle=True)
            header = np.load(os.path.join(info['output_path'], 'arrays', info['subject_path'], 'dyn_header_15.npz'), allow_pickle=True)
            timepoints = data['timepoints']
            fa = data['fa']
            weight = header['weight']
            scan_date = header['study_date'] #both scans should have the same date

            if scan == 1:
                Tr1 = data['Tr']
            elif scan == 2:
                Tr2 = data['Tr']

            df = pd.DataFrame({'time': timepoints[:70], 'fa':fa, 'liver': liver_signal, 'liver-2':'', 'aorta': aorta_signal[:70], 'portal-vein':'', 'kidney':'', 'noise':'', 'fat':'', 'spleen':'', 'vena-cava':'', 'gallbladder':'', 'left-ventricle':'', 'right-ventricle':'', 'muscle':'', 'lung':'', 'vertebra':'', 'intestine':''})
            df.to_excel(writer, sheet_name=f'dyn{scan}', index=False)
        
        # save MOLLI T1 values
        # TO DO: add extraction from segmentation into corresponding .npz files
        # each MOLLI should have npz keys: time,liver,aorta, spleen, portal vein, vena-cava, fat, gallbladder, left-ventricle, right-ventricle, kidney-parenchyma, muscle, lung, vertebra, intestine
        # only need T1 values for time, liver and aorta
        molli_1 = np.load(os.path.join(output_path, 'molli', f'S{subject}_V{visit}_S1_prec_molli_ROIvalues.npz'))
        molli_2 = np.load(os.path.join(output_path, 'molli', f'S{subject}_V{visit}_S1_postc_molli_ROIvalues.npz'))
        molli_3 = np.load(os.path.join(output_path, 'molli', f'S{subject}_V{visit}_S2_prec_molli_ROIvalues.npz'))

        i=1
        for molli in [molli_1, molli_2, molli_3]:
            molli_dict = {
                'time': molli['time'],
                'liver': molli['liver'],
                'aorta': molli['aorta'],
                'spleen': '',
                'portal vein': '',
                'vena-cava': '',
                'fat': '',
                'gallbladder': '',
                'left-ventricle': '',
                'right-ventricle': '',
                'kidney-parenchyma': '',
                'muscle': '',
                'lung': '',
                'vertebra': '',
                'intestine': ''
            }
            df = pd.DataFrame([molli_dict])
            df.to_excel(writer, sheet_name=f'MOLLI{i}', index=False)
            i+=1

        #TO DO implement extraction of constants from segmentations (liver volumes)
        # using placeholder values for now
        constants = {
            'weight': weight,
            'dose1': 0.025, # not in dicom use default
            'dose2': 0.025, # not in dicom use default
            'baseline': 60, # TO DO: not in dicom. use default for scan number. NOTE: the baseline time are different between scans
            'scan-date': scan_date,
            'liver-volume-voxels-n': 10273, # TO DO: extract from segmentation. mask array files and spacing from combined dynamic- only 1 value across scans propose average
            'liver-volume-mm3': 829195.4964, # TO DO: extract from segmentation
            'kidney-volume-voxels-n': None,
            'kidney-volume-mm3': None,
            'tr-first-dynamic': Tr1,
            'tr-second-dynamic': Tr1 
        }
        
        df = pd.DataFrame(list(constants.items()), columns=['name', 'value'])
        df.to_excel(writer, sheet_name='const', index=False)

        print(f'Subject{subject}_Visit{visit}_Scan{scan} AIF saved')
   