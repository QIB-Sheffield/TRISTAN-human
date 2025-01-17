import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import os
from utils import get_subject_visits, check_dirs_exist, save_as_nifti

rel_path = os.path.dirname(__file__)
subject_visits_path = os.path.join(rel_path, '..\\data\\subjects_visits.txt')
data_path = os.path.join(rel_path,'..\\outputs\\np_arrays')
save_path = os.path.join(rel_path,'..\\outputs\\nifti')

# Read volunteers and visits from the text file
subjects, visits, scans = get_subject_visits(subject_visits_path)

check_dirs_exist(save_path)

subject = subjects[0]
visit = visits[0]
scan = scans[0]

subjects = [1]
visits = [1]
scans = [1,2]

for subject in subjects:
    for visit in visits:
        for scan in scans:
            if subject == 3:
                acq_type = '[SPGR_cor_fb_fa12_dynamic_4.5_os]'
            else:
                acq_type = '[None]'
            
            series_no = [ [['Series 1101 ', 'Series 901 '],['Series 901 ', 'Series 1001 ']] , [['Series 1001 ', 'Series 1101 '],['Series 901 ', 'Series 1001 ']] , [['Series 1001 ','Series 901 '], ['Series 901 ', 'Series 901 ']]]

            data = np.load("{}\\Subject_{}\\Visit_{}\\Scan_{}\\{}{}.npy".format(data_path,subject,visit,scan,series_no[subject-1][visit-1][scan-1],acq_type),allow_pickle=True)
            time = np.load("{}\\Subject_{}\\Visit_{}\\Scan_{}\\time_array.npy".format(data_path,subject,visit,scan),allow_pickle=True)
            corrected_time = time - time[0]

            print("{}\\Subject_{}\\Visit_{}\\Scan_{}\\{}{}.npy".format(data_path,subject,visit,scan,series_no[subject-1][visit-1][scan-1],acq_type))
            print(save_path)
            print(f'subject{subject}_visit{visit}_scan{scan}_full_REPEAT.nii.gz')

            # Create a figure and axes for the plot
            fig, ax = plt.subplots()

            # Create initial plot with the first time and slice
            time_index = 0
            slice_index = 0
            image = ax.imshow(data[..., slice_index, time_index], vmin=0, vmax=np.percentile(data,95), cmap='gray')

            # Create sliders for time and slice selection
            ax_time_slider = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')
            ax_slice_slider = plt.axes([0.25, 0.05, 0.65, 0.03], facecolor='lightgoldenrodyellow')

            time_slider = Slider(ax_time_slider, 'Time', 0, data.shape[3] - 1, valinit=0, valstep=1)
            slice_slider = Slider(ax_slice_slider, 'Slice', 0, data.shape[2] - 1, valinit=0, valstep=1)

            # Define the update function for the sliders
            def update(val):
                time_index = int(time_slider.val)
                slice_index = int(slice_slider.val)
                image.set_data(data[:,:, slice_index, time_index])
                fig.canvas.draw_idle()

            # Connect the update function to the sliders
            time_slider.on_changed(update)
            slice_slider.on_changed(update)

            # Show the plot
            plt.show()#

            # Get the initial signal
            initial_signal = data[..., 0]
            enhancement = data - initial_signal[..., np.newaxis]

            # Get the maximum enhancement
            max_enhance = np.max(enhancement, axis=-1)

            # Get the time to maximum signal
            ttp = np.argmax(data, axis=3)
            ttp = corrected_time[ttp]

            ttp=ttp/np.max(ttp)
            ttp=1-ttp

            #save_as_nifti(max_enhance, save_path, f'subject{subject}_visit{visit}_scan{scan}_maxEnhance.nii.gz')
            save_as_nifti(data, save_path, f'subject{subject}_visit{visit}_scan{scan}_full_REPEAT.nii.gz')
            #save_as_nifti(ttp, save_path, f'subject{subject}_visit{visit}_scan{scan}_TTP.nii.gz')

            print(f"Exported NiFTi for subject {subject}, visit {visit}, scan {scan}")