import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import nibabel as nib
import nrrd

show_aif_plot = False
save_aif = True

# Set relative path
rel_path = os.path.dirname(__file__)

# Construct all relevant paths
data_path =  os.path.join(rel_path,'..\\outputs\\np_arrays')
aif_path = os.path.join(rel_path,'..\\outputs\\aifs')
save_path = os.path.join(rel_path,'..\\outputs\\figures')
seg_path = os.path.join(rel_path,'..\\outputs\\segmentation\\aif')
subject_visits_path = os.path.join(rel_path, '..\\data\\subjects_visits.txt')

# Read volunteers and visits from the text file
with open(subject_visits_path, 'r') as file:
    lines = file.readlines()
    subjects = list(map(int, lines[0].strip().split(': ')[1].split(',')))
    visits = list(map(int, lines[1].strip().split(': ')[1].split(',')))
    scans = list(map(int, lines[2].strip().split(': ')[1].split(',')))

os.makedirs(save_path, exist_ok=True)
os.makedirs(aif_path, exist_ok=True)

          
series_no = [ [['Series 1101 ', 'Series 901 '],['Series 901 ', 'Series 1001 ']] , [['Series 1001 ', 'Series 1101 '],['Series 901 ', 'Series 1001 ']] , [['Series 1001 ','Series 901 '], ['Series 901 ', 'Series 901 ']]]

subjects = [1,2]
visits = [1,2]
scans = [1,2]

for subject in subjects:
    for visit in visits:
        for scan in scans:
            
            if subject == 3:
                acq_type = '[SPGR_cor_fb_fa12_dynamic_4.5_os]'
            else:
                acq_type = '[None]'

            # Load the data
            data = np.load("{}\\Subject_{}\\Visit_{}\\Scan_{}\\{}{}.npy".format(data_path,subject,visit,scan,series_no[subject-1][visit-1][scan-1],acq_type),allow_pickle=True)

            seg_file_name = f"s{subject}_v{visit}_s{scan}_aorta_ME.nii.gz"

            if '.nrrd' in seg_file_name:
                mask, _ = nrrd.read(f"{seg_path}\\{seg_file_name}")
            elif '.nii' in seg_file_name:
                seg_file = nib.load(f"{seg_path}\\{seg_file_name}")
                # Get the data from the NIfTI file
                mask = seg_file.get_fdata()

            mask = mask[..., np.newaxis]

            # Apply the mask to the data
            roi_data = data * mask

            # Calculate the mean of the 2D ROI across the 4th dimension (time)
            aif_signal = np.sum(roi_data, axis=(0,1,2))/np.sum(mask)

            # Assuming time, aif_signal, and baseline are 1D numpy arrays with the same length as roi_data
            time = np.load(f'{data_path}\\Subject_{subject}\\Visit_{visit}\\Scan_{scan}\\time_array.npy')
            
            if scan == 1:
                baseline = int((1*50)/1.8)
            elif scan == 2:
                baseline = int((1*590)/1.8)

            pixel_spacing = np.load(f'{data_path}\\Subject_{subject}\\Visit_{visit}\\Scan_{scan}\\pixelspacing.npy')
            slice_thickness = np.load(f'{data_path}\\Subject_{subject}\\Visit_{visit}\\Scan_{scan}\\slice_thickness.npy')
            adj_time = time-time[0]

             # Plot the AIF signal
            if show_aif_plot:
                plt.plot(adj_time, aif_signal)
                plt.show()
            else:
                pass

            if save_aif:
                # Save the AIF signal, adjusted time, baseline, pixel spacing, and slice thickness to a .npz file
                np.savez_compressed(f'{aif_path}\\Subject{subject}_Visit{visit}_Scan{scan}_aif.npz', aif_signal=aif_signal, adj_time=adj_time, baseline=baseline, pixel_spacing=pixel_spacing, slice_thickness=slice_thickness)
                
                # Create a DataFrame with time and aif_signal as columns
                df = pd.DataFrame({'time': time, 'aorta': aif_signal})

                # Save the DataFrame to a CSV file
                df.to_csv(f'{aif_path}\\Subject{subject}_Visit{visit}_Scan{scan}_aif.csv', index=False)
                
                print(f'Subject{subject}_Visit{visit}_Scan{scan} AIF saved')
            else:
                pass