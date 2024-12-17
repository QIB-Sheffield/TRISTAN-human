import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import nibabel as nib
import nrrd

from utils import get_subject_visits, check_dirs_exist
show_plot = False
save_signals = True

# Set relative path
rel_path = os.path.dirname(__file__)

# Construct all relevant paths
data_path =  os.path.join(rel_path,'..\\outputs\\np_arrays')
seg_path_a = os.path.join(rel_path,'..\\outputs\\segmentation\\aif')
seg_path_l = os.path.join(rel_path,'..\\outputs\\segmentation\\liver')
subject_visits_path = os.path.join(rel_path, '..\\data\\subjects_visits.txt')

results_path = os.path.join(rel_path, '..\\results\\csv')
results_path_v1 = os.path.join(results_path, 'visit1')
results_path_v2 = os.path.join(results_path, 'visit2')

subjects, visits, scans = get_subject_visits(subject_visits_path)

check_dirs_exist(results_path, results_path_v1, results_path_v2)

          
series_no = [ [['Series 1101 ', 'Series 901 '],['Series 901 ', 'Series 1001 ']] , [['Series 1001 ', 'Series 1101 '],['Series 901 ', 'Series 1001 ']] , [['Series 1001 ','Series 901 '], ['Series 901 ', 'Series 901 ']]]

subjects = [1]
visits = [1]
scans = [1,2]
r_paths = [results_path_v1, results_path_v2]

for subject in subjects:
    for visit in visits:
        with pd.ExcelWriter(f'{r_paths[visit-1]}\\00{subject}-visit{visit}.xlsx', engine='xlsxwriter') as writer:
            for scan in scans:
                
                if subject == 3:
                    acq_type = '[SPGR_cor_fb_fa12_dynamic_4.5_os]'
                else:
                    acq_type = '[None]'

                # Load the data
                data = np.load("{}\\Subject_{}\\Visit_{}\\Scan_{}\\{}{}.npy".format(data_path,subject,visit,scan,series_no[subject-1][visit-1][scan-1],acq_type),allow_pickle=True)

                aif_seg_file_name = f"s{subject}_v{visit}_s{scan}_aorta_ME.nii.gz"
                liver_seg_file_name = f"Subject{subject}_Visit{visit}_Scan{scan}_coreg_AUC_liver.nrrd"

                if '.nrrd' in aif_seg_file_name:
                    aif_mask, _ = nrrd.read(f"{seg_path_a}\\{aif_seg_file_name}")
                elif '.nii' in aif_seg_file_name:
                    seg_file = nib.load(f"{seg_path_a}\\{aif_seg_file_name}")
                    # Get the data from the NIfTI file
                    aif_mask = seg_file.get_fdata()

                if '.nrrd' in liver_seg_file_name:
                    liver_mask, _ = nrrd.read(f"{seg_path_l}\\{liver_seg_file_name}")
                elif '.nii' in liver_seg_file_name:
                    seg_file = nib.load(f"{seg_path_l}\\{liver_seg_file_name}")
                    # Get the data from the NIfTI file
                    liver_mask = seg_file.get_fdata()

                aif_mask = aif_mask[..., np.newaxis]
                liver_mask = liver_mask[..., np.newaxis]

                # Apply the mask to the data
                aif_roi_data = data * aif_mask
                liver_roi_data = data * liver_mask

                # Calculate the mean of the 3D ROI across the 4th dimension (time)
                aif_signal = np.sum(aif_roi_data, axis=(0,1,2))/np.sum(aif_mask)

                # Calculate the mean of the 3D ROI across the 4th dimension (time)
                liver_signal = np.sum(liver_roi_data, axis=(0,1,2))/np.sum(liver_mask)

                # Assuming time, aif_signal, and baseline are 1D numpy arrays with the same length as roi_data
                time = np.load(f'{data_path}\\Subject_{subject}\\Visit_{visit}\\Scan_{scan}\\time_array.npy')
                
            
                # Plot the AIF signal
                if show_plot:
                    plt.plot(time-time[0], aif_signal)
                    plt.plot(time-time[0], liver_signal)
                    plt.show()
                else:
                    pass

                if save_signals:
                    # Create a DataFrame with time and aif_signal as columns
                    df = pd.DataFrame({'time': time, 'fa':12, 'liver': liver_signal, 'liver-2':'', 'aorta': aif_signal, 'portal-vein':'', 'kidney':'', 'noise':'', 'fat':'', 'spleen':'', 'vena-cava':'', 'gallbladder':'', 'left-ventricle':'', 'right-ventricle':'', 'muscle':'', 'lung':'', 'vertebra':'', 'intestine':''})

                    # Save the DataFrame to a CSV file
                    df.to_excel(writer, sheet_name=f'dyn{scan}', index=False)
                    
                    print(f'Subject{subject}_Visit{visit}_Scan{scan} AIF saved')
                else:
                    pass
